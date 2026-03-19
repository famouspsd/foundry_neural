def create_safe_pdf(text):
    # This line is the 'Magic Fix'
    # It converts symbols it doesn't know into '?' or safe versions
    clean_text = text.encode('latin-1', 'replace').decode('latin-1')
    
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.multi_cell(0, 10, txt=clean_text)
    return pdf.output()
    

            
