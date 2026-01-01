import streamlit as st
import google.generativeai as genai
import fitz  # PyMuPDF
import io
from PIL import Image
import json
import os

# =============================
# STREAMLIT CONFIG
# =============================
st.set_page_config(
    page_title="Invoice OCR using Gemini",
    layout="wide"
)

st.title("📄 Invoice OCR & Extraction (Gemini)")
st.write("Upload an invoice PDF to extract structured JSON data.")

# =============================
# GEMINI SETUP
# =============================
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# =============================
# KEY–VALUE PAIR SCHEMA
# =============================
INVOICE_SCHEMA = """
Return ONLY valid JSON using this schema.
If a field is missing, use null.
Dates must be YYYY-MM-DD.
Numbers must be numbers (not strings).

{
  "seller_details": {
    "seller_name": null,
    "seller_address": null,
    "seller_city": null,
    "seller_state": null,
    "seller_pan": null,
    "seller_gst": null
  },
  "buyer_details": {
    "buyer_name": null,
    "buyer_address": null,
    "buyer_city": null,
    "buyer_state": null,
    "buyer_pan": null,
    "buyer_gst": null
  },
  "invoice_details": {
    "invoice_number": null,
    "invoice_value": null,
    "invoice_date": null,
    "invoice_currency": "INR"
  },
  "item_details": [
    {
      "item_name": null,
      "item_description": null,
      "item_hsn_sac_code": null,
      "item_quantity": null,
      "item_rate": null,
      "item_value": null,
      "cgst_rate": null,
      "cgst_amount": null,
      "sgst_rate": null,
      "sgst_amount": null,
      "igst_rate": null,
      "igst_amount": null
    }
  ]
}
"""

# =============================
# PDF → IMAGE (PyMuPDF)
# =============================
def pdf_to_images(pdf_bytes):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    images = []

    for page in doc:
        pix = page.get_pixmap(dpi=300)
        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes))
        images.append(image)

    return images

# =============================
# OCR FUNCTIONS
# =============================
def extract_from_image(image: Image.Image):
    response = model.generate_content(
        [
            "You are an expert invoice OCR and accounting assistant.",
            INVOICE_SCHEMA,
            image
        ],
        generation_config={
            "temperature": 0,
            "response_mime_type": "application/json"
        }
    )
    return response.text


def extract_invoice_from_pdf(pdf_bytes):
    images = pdf_to_images(pdf_bytes)

    # Process first page (most invoices are single-page)
    raw_json = extract_from_image(images[0])
    return json.loads(raw_json)

# =============================
# UI
# =============================
uploaded_file = st.file_uploader(
    "Upload Invoice PDF",
    type=["pdf"]
)

if uploaded_file and st.button("🚀 Extract Invoice"):
    with st.spinner("Processing invoice..."):
        result = extract_invoice_from_pdf(uploaded_file.read())

        st.subheader("📦 Extracted JSON")
        st.json(result)

        st.download_button(
            label="⬇️ Download JSON",
            data=json.dumps(result, indent=2),
            file_name="invoice.json",
            mime="application/json"
        )
