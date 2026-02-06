import os
import json

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib.enums import TA_LEFT, TA_JUSTIFY
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


def create_pdf_from_summary(summary_file_path, output_pdf_path):
    if not REPORTLAB_AVAILABLE:
        return False
    
    with open(summary_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    summary_text = data.get("output", "")
    
    if not summary_text:
        return False
    
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18
    )
    
    story = []
    
    styles = getSampleStyleSheet()
    
    try:
        font_paths = [
            '/System/Library/Fonts/Supplemental/Times New Roman.ttf',
            '/System/Library/Fonts/Times New Roman.ttf',
            '/Library/Fonts/Times New Roman.ttf',
            '/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf',
            '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf',
            'C:/Windows/Fonts/times.ttf',
            'C:/Windows/Fonts/timesbd.ttf',
            'C:/Windows/Fonts/timesi.ttf',
        ]
        
        times_found = False
        for font_path in font_paths:
            if os.path.exists(font_path):
                try:
                    pdfmetrics.registerFont(TTFont('Times-Roman', font_path))
                    times_found = True
                    break
                except:
                    continue
        
        if times_found:
            normal_style = ParagraphStyle(
                'CustomNormal',
                parent=styles['Normal'],
                fontName='Times-Roman',
                fontSize=12,
                leading=14,
                alignment=TA_LEFT,
                spaceAfter=12
            )
        else:
            normal_style = ParagraphStyle(
                'CustomNormal',
                parent=styles['Normal'],
                fontSize=12,
                leading=14,
                alignment=TA_LEFT,
                spaceAfter=12
            )
    except Exception as e:
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontSize=12,
            leading=14,
            alignment=TA_LEFT,
            spaceAfter=12
        )
    
    paragraphs = summary_text.split('\n\n')
    
    if len(paragraphs) == 1:
        paragraphs = summary_text.split('\n')
    
    for para_text in paragraphs:
        if para_text.strip():
            para_text = ' '.join(para_text.split('\n'))
            para_text = para_text.replace('&', '&amp;')
            para_text = para_text.replace('<', '&lt;')
            para_text = para_text.replace('>', '&gt;')
            para_text = para_text.replace('"', '&quot;')
            para_text = para_text.replace("'", '&apos;')
            
            story.append(Paragraph(para_text, normal_style))
            story.append(Spacer(1, 0.15 * inch))
    
    try:
        doc.build(story)
        return True
    except Exception as e:
        return False


def generate_pdfs_for_summaries(summaries_dir, output_pdf_dir):
    from tqdm import tqdm
    
    os.makedirs(output_pdf_dir, exist_ok=True)
    
    attack_dirs = [d for d in os.listdir(summaries_dir) 
                   if os.path.isdir(os.path.join(summaries_dir, d)) and not d == "pdf"]
    
    if attack_dirs:
        for attack_name in tqdm(attack_dirs, desc="Generating PDFs"):
            attack_dir = os.path.join(summaries_dir, attack_name)
            attack_pdf_dir = os.path.join(output_pdf_dir, attack_name)
            os.makedirs(attack_pdf_dir, exist_ok=True)
            
            subgraph_dirs = [d for d in os.listdir(attack_dir) 
                           if os.path.isdir(os.path.join(attack_dir, d)) and 
                           (d.startswith("subgraph_") or d.startswith("magic_subgraph_") or d.startswith("split_"))]
            
            for subgraph_dir in sorted(subgraph_dirs):
                subgraph_path = os.path.join(attack_dir, subgraph_dir)
                summary_file = os.path.join(subgraph_path, "summary.txt")
                labelled_summary_file = os.path.join(subgraph_path, "labelled_summary.txt")
                
                subgraph_pdf_dir = os.path.join(attack_pdf_dir, subgraph_dir)
                os.makedirs(subgraph_pdf_dir, exist_ok=True)
                
                if os.path.exists(summary_file):
                    pdf_filename = "summary.pdf"
                    pdf_path = os.path.join(subgraph_pdf_dir, pdf_filename)
                    create_pdf_from_summary(summary_file, pdf_path)
                
                if os.path.exists(labelled_summary_file):
                    pdf_filename = "labelled_summary.pdf"
                    pdf_path = os.path.join(subgraph_pdf_dir, pdf_filename)
                    create_pdf_from_summary(labelled_summary_file, pdf_path)
    else:
        summary_files = []
        labelled_summary_files = []
        
        for filename in os.listdir(summaries_dir):
            if filename.startswith("summary_") and filename.endswith(".txt") and not filename.startswith("labelled_"):
                summary_files.append(filename)
            elif filename.startswith("labelled_summary_") and filename.endswith(".txt"):
                labelled_summary_files.append(filename)
        
        for filename in tqdm(sorted(summary_files), desc="Generating PDFs"):
            summary_path = os.path.join(summaries_dir, filename)
            pdf_filename = filename.replace(".txt", ".pdf")
            pdf_path = os.path.join(output_pdf_dir, pdf_filename)
            create_pdf_from_summary(summary_path, pdf_path)
        
        for filename in tqdm(sorted(labelled_summary_files), desc="Generating labelled PDFs"):
            summary_path = os.path.join(summaries_dir, filename)
            pdf_filename = filename.replace(".txt", ".pdf")
            pdf_path = os.path.join(output_pdf_dir, pdf_filename)
            create_pdf_from_summary(summary_path, pdf_path)


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        summaries_dir = sys.argv[1]
        output_pdf_dir = sys.argv[2]
        generate_pdfs_for_summaries(summaries_dir, output_pdf_dir)

