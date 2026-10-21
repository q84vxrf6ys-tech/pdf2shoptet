import io
import json
import os
import pandas as pd
import pdfplumber
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="PDF2Shoptet - Převodník ceníků", layout="wide")

st.title("📄 PDF2Shoptet Converter")
st.write("Nahrajte PDF ceník od dodavatele a stáhněte si strukturované CSV pro Shoptet.")

# Boční panel pro API klíč
with st.sidebar:
    st.header("Nastavení")
    api_key = st.text_input("OpenAI API Key", type="password")
    vat_default = st.selectbox("Výchozí DPH (%)", [21, 12, 0], index=0)

uploaded_file = st.file_uploader("Vyberte PDF ceník", type=["pdf"])

def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text

def parse_pdf_with_ai(text_content, api_key):
    client = OpenAI(api_key=api_key)
    
    prompt = f"""
    Jsi expertní datový parser. Z následujícího textu z PDF ceníku vytáhni všechny produkty.
    Pro každý produkt vytáhni tyto parametry:
    - code (kód produktu nebo EAN; pokud chybí, vygeneruj např. KOD1, KOD2...)
    - name (přesný název produktu)
    - price (cena bez DPH jako číslo; pokud nelze určit, dej 0)
    - stock (skladová zásoba jako číslo, pokud není uvedeno, dej 10)

    Pokud v textu nejsou žádné produkty nebo položky ceníku, vrať prázdné pole [].
    Vrať výhradně čistý JSON array objektů bez jakéhokoliv dalšího textu nebo Markdownu.
    
    Příklad formátu:
    [
      {{"code": "ABC123", "name": "Produkt X", "price": 150.50, "stock": 10}}
    ]

    Text k analýze:
    {text_content[:12000]}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1
    )
    
    raw_content = response.choices[0].message.content.strip()
    if raw_content.startswith("```json"):
        raw_content = raw_content[7:-3].strip()
    elif raw_content.startswith("```"):
        raw_content = raw_content[3:-3].strip()
        
    return json.loads(raw_content)

if uploaded_file and api_key:
    if st.button("Zpracovat PDF", type="primary"):
        with st.spinner("Čtu PDF a extrahuji data pomocí AI..."):
            try:
                pdf_text = extract_text_from_pdf(uploaded_file)
                if not pdf_text.strip():
                    st.error("V PDF nebyl nalezen žádný čitelný text (může jít o naskenovaný obrázek bez OCR).")
                else:
                    parsed_data = parse_pdf_with_ai(pdf_text, api_key)
                    
                    if not parsed_data:
                        st.warning("AI v nahraném dokumentu nenašla žádné produkty/položky ceníku.")
                    else:
                        df = pd.DataFrame(parsed_data)
                        
                        # Bezpečné doplnění chybějících sloupců
                        for col in ["code", "name", "price", "stock"]:
                            if col not in df.columns:
                                df[col] = "" if col in ["code", "name"] else 0
                        
                        # Namapování na povinné sloupce pro Shoptet import
                        shoptet_df = pd.DataFrame({
                            "code": df["code"],
                            "pairCode": "",
                            "name": df["name"],
                            "price": df["price"],
                            "standardPrice": df["price"],
                            "vat": vat_default,
                            "amount": df["stock"],
                            "unit": "ks",
                            "visible": 1
                        })
                        
                        st.success(f"Úspěšně zpracováno {len(shoptet_df)} položek!")
                        st.subheader("Náhled vygenerovaných dat pro Shoptet")
                        st.dataframe(shoptet_df)
                        
                        # Generování CSV v kódování UTF-8 s window-friendly oddělovačem
                        csv_buffer = io.StringIO()
                        shoptet_df.to_csv(csv_buffer, index=False, sep=";")
                        
                        st.download_button(
                            label="📥 Stáhnout Shoptet CSV",
                            data=csv_buffer.getvalue().encode('utf-8-sig'),
                            file_name="shoptet_import.csv",
                            mime="text/csv"
                        )
            except Exception as e:
                st.error(f"Chyba při zpracování: {str(e)}")
elif uploaded_file and not api_key:
    st.warning("Pro zpracování zadejte OpenAI API klíč v levém menu.")
