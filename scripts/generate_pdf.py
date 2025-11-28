
import markdown
from xhtml2pdf import pisa
import os

def convert_md_to_pdf(source_md, output_pdf):
    # Read Markdown
    with open(source_md, 'r', encoding='utf-8') as f:
        text = f.read()

    # Convert to HTML
    # Enable extensions for tables and fenced code blocks
    html = markdown.markdown(text, extensions=['tables', 'fenced_code'])

    # Add some basic CSS for styling
    css = """
    <style>
        body { font-family: Helvetica, sans-serif; font-size: 10pt; }
        h1 { color: #2c3e50; font-size: 24pt; border-bottom: 2px solid #2c3e50; padding-bottom: 10px; }
        h2 { color: #34495e; font-size: 18pt; margin-top: 20px; }
        h3 { color: #7f8c8d; font-size: 14pt; margin-top: 15px; }
        code { background-color: #f4f4f4; padding: 2px; font-family: Courier; }
        pre { background-color: #f8f9fa; padding: 10px; border: 1px solid #e9ecef; white-space: pre-wrap; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        img { max-width: 100%; height: auto; margin: 20px 0; }
    </style>
    """
    
    full_html = f"<html><head>{css}</head><body>{html}</body></html>"

    # Write HTML for debug (optional)
    # with open("debug.html", "w") as f:
    #     f.write(full_html)

    # Convert to PDF
    with open(output_pdf, "wb") as result_file:
        pisa_status = pisa.CreatePDF(
            full_html,
            dest=result_file,
            encoding='utf-8'
        )

    if pisa_status.err:
        print(f"Error converting to PDF: {pisa_status.err}")
        return False
    
    print(f"Successfully created {output_pdf}")
    return True

if __name__ == "__main__":
    source = "PROJECT_DOCUMENTATION.md"
    output = "PROJECT_DOCUMENTATION.pdf"
    
    if os.path.exists(source):
        convert_md_to_pdf(source, output)
    else:
        print(f"Source file {source} not found!")
