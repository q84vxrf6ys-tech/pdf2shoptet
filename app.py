import io
import json
import re
import pandas as pd
import pdfplumber
import streamlit as st
from openai import OpenAI

# Nastavení stránky a ikony
st.set_page_config(
    page_title="PDF2Shoptet | Automatický převodník ceníků",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Vlastní CSS pro moderní vzhled
st.markdown("""
    <style>
    .main {
        background-color: #f8fafc;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .badge-container {
        display: flex;
        gap: 10px;
        margin-bottom: 20px;
    }
    .badge {
        background-color: #e0f2fe;
        color: #0369a1;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# Boční panel nastavení
with st.sidebar:
    st.image("https://img.icons8.com/isometric/96/flash-on.png", width=60)
    st.title("PDF2Shoptet SaaS")
    st.caption("v1.2 Commercial Release")
    st.markdown("---")
    
    st.subheader("🔑 Konfigurace")
    api_key = st.text_input("OpenAI API Key", type="password", help="Vložte váš API klíč ze služby OpenAI (platform.openai.com)")
    vat_default = st.selectbox("Výchozí sazba DPH (%)", [21, 12, 0], index=0)
    separator = st.selectbox("Oddělovač CSV", ["; (Standard Shoptet)", ", (Čárka)"], index=0)
    
    st.markdown("---")
    st.info("💡 **Tip pro e-shopy:** Získaný CSV soubor nahrajte v administraci Shoptetu přes *Produkty -> Import*.")

# Hlavní hlavička
st.markdown('<div class="hero-title">⚡ PDF2Shoptet Converter</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Přetavte jakýkoliv PDF ceník nebo katalog od dodavatele do čistého Shoptet CSV během 5 sekund.</div>', unsafe_allow_html=True)

st.markdown("""
    <div class="badge-container">
        <span class="badge">✓ 100% Shoptet Ready</span>
        <span class="badge">✓ Detekce EAN / Kódů</span>
        <span class="badge">✓ Automatické čištění cen</span>
    </div>
""", unsafe_allow_html=True)

# Nahrávání souboru
uploaded_file = st.file_uploader("📂 Nahrajte PDF ceník nebo katalog", type=["pdf"], help="Podporované formáty: PDF ceníky, faktury a katalogy.")

def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
    return text

def clean_json_response(raw_response):
    raw_content = raw_response.strip()
    if raw_content.startswith("```json"):
        raw_content = raw_content[7:]
    if raw_content.endswith("```"):
        raw_content = raw_content[:-3]
    return raw_content.strip()

def parse_pdf_with_ai(text_content, api_key):
    client = OpenAI(api_key=api_key)
    
    prompt = f"""
    Jsi specializovaný datový inženýr pro e-commerce. Z níže uvedeného textu z PDF ceníku vytáhni všechny samostatné produkty.
    
    Pravidla pro extrakci:
    1. "code": Kód produktu, EAN nebo SKU. Pokud v PDF chybí, vygeneruj unikátní kód podle názvu (např. PRD-001).
    2. "name": Přesný, plný název produktu.
    3. "price": Cena bez DPH jako ČISTÉ ČÍSLO (převeď řetězce typu "1 890 Kč", "120,50 EUR" na čisté číslo jako 1890 nebo 120.5). Odstraň měny a mezery.
    4. "stock": Skladová zásoba (číslo). Pokud chybí nebo je text typu "skladem", uveď 10.

    Odpověď MUSÍ být výhradně validní JSON pole objektů bez jakéhokoliv dalšího úvodního nebo závěrečného textu.

    Příklad formátu:
    [
        {{"code": "EL-102", "name": "Herní myš", "price": 450.0, "stock": 10}}
    ]

    Text k analýze:
    {text_content[:20000]}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    
    cleaned = clean_json_response(response.choices[0].message.content)
    return json.loads(cleaned)

if uploaded_file:
    if not api_key:
        st.error("⚠️ Pro zpracování PDF zadejte v levém menu váš OpenAI API klíč.")
    else:
        if st.button("🚀 Zpracovat ceník a vygenerovat CSV", type="primary", use_container_width=True):
            with st.spinner("⏳ Čtu PDF dokument a extrahuji strukturovaná data pomocí AI..."):
                try:
                    pdf_text = extract_text_from_pdf(uploaded_file)
                    
                    if not pdf_text.strip():
                        st.error("❌ V dokumentu nebyl nalezen žádný text. Pokud jde o naskenovaný obrázek, převeďte jej nejprve pomocí OCR.")
                    else:
                        parsed_data = parse_pdf_with_ai(pdf_text, api_key)
                        
                        if not parsed_data:
                            st.warning("⚠️ V dokumentu nebyly nalezeny žádné produktové položky.")
                        else:
                            df = pd.DataFrame(parsed_data)
                            
                            # Ošetření chybějících sloupců
                            for col in ["code", "name", "price", "stock"]:
                                if col not in df.columns:
                                    df[col] = "" if col in ["code", "name"] else 0
                            
                            # Převod cen a zásob na správné datové typy
                            df["price"] = pd.to_numeric(df["price"], errors="coerce").fillna(0)
                            df["stock"] = pd.to_numeric(df["stock"], errors="coerce").fillna(10).astype(int)

                            # Namapování do shoptetového DataFrame
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
                            
                            # Zobrazení analytických metrik
                            col1, col2, col3 = st.columns(3)
                            col1.metric("Nalezeno produktů", f"{len(shoptet_df)} ks")
                            col2.metric("Průměrná cena", f"{shoptet_df['price'].mean():.2f} Kč")
                            col3.metric("Celková hodnota zásoby", f"{(shoptet_df['price'] * shoptet_df['amount']).sum():,.0f} Kč")

                            st.markdown("### 📊 Náhled vygenerovaného ceníku")
                            st.dataframe(shoptet_df, use_container_width=True)
                            
                            # Generování CSV
                            sep_char = ";" if ";" in separator else ","
                            csv_buffer = io.StringIO()
                            shoptet_df.to_csv(csv_buffer, index=False, sep=sep_char)
                            
                            clean_filename = uploaded_file.name.replace(".pdf", "").replace(" ", "_")
                            st.download_button(
                                label="📥 Stáhnout kompletní Shoptet CSV soubor",
                                data=csv_buffer.getvalue().encode('utf-8-sig'),
                                file_name=f"shoptet_import_{clean_filename}.csv",
                                mime="text/csv",
                                type="primary",
                                use_container_width=True
                            )
                except Exception as e:
                    st.error(f"❌ Chyba při zpracování: {str(e)}")
