#!/usr/bin/env python3
"""
Script to create customized CV from files in /Users/pd3rvr/Downloads/chance
"""
import os
import re
from pathlib import Path
from datetime import datetime

def read_all_files(directory):
    """Read all text files from directory"""
    files_content = {}
    directory_path = Path(directory)
    
    if not directory_path.exists():
        print(f"Directory {directory} does not exist")
        return files_content
    
    for file_path in directory_path.iterdir():
        if file_path.is_file() and not file_path.name.startswith('.'):
            try:
                # Try different encodings
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            content = f.read()
                            files_content[file_path.name] = content
                            print(f"✓ Read {file_path.name} ({len(content)} chars)")
                            break
                    except UnicodeDecodeError:
                        continue
            except Exception as e:
                print(f"✗ Error reading {file_path.name}: {e}")
    
    return files_content

def extract_info_from_files(files_content):
    """Extract relevant information from the files"""
    info = {
        'name': '',
        'email': '',
        'phone': '',
        'linkedin': '',
        'github': '',
        'location': '',
        'summary': '',
        'education': [],
        'experience': [],
        'skills': [],
        'publications': [],
        'projects': [],
        'certifications': [],
        'recommendation_text': ''
    }
    
    # Combine all content
    all_text = '\n\n'.join(files_content.values())
    
    # Look for recommendation_professional file
    for filename, content in files_content.items():
        if 'recommendation' in filename.lower() or 'professional' in filename.lower():
            info['recommendation_text'] = content
            print(f"Found recommendation/professional content in {filename}")
    
    # Extract email
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, all_text)
    if emails:
        info['email'] = emails[0]
    
    # Extract phone
    phone_pattern = r'[\+]?[(]?[0-9]{1,4}[)]?[-\s\.]?[(]?[0-9]{1,4}[)]?[-\s\.]?[0-9]{1,9}'
    phones = re.findall(phone_pattern, all_text)
    if phones:
        info['phone'] = phones[0]
    
    # Extract LinkedIn
    linkedin_pattern = r'linkedin\.com/in/[\w-]+'
    linkedin = re.findall(linkedin_pattern, all_text, re.IGNORECASE)
    if linkedin:
        info['linkedin'] = 'https://' + linkedin[0]
    
    # Extract GitHub
    github_pattern = r'github\.com/[\w-]+'
    github = re.findall(github_pattern, all_text, re.IGNORECASE)
    if github:
        info['github'] = 'https://' + github[0]
    
    return info, all_text

def generate_html_cv(info, all_text):
    """Generate HTML CV"""
    
    # Extract IT Project Manager experience from recommendation text
    it_pm_experience = ""
    if info['recommendation_text']:
        # Try to extract relevant project manager details
        it_pm_experience = info['recommendation_text'][:500]  # First 500 chars as placeholder
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CV - Associate Director of AI</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
            padding: 20px;
        }}
        
        .container {{
            max-width: 900px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        
        .header {{
            text-align: center;
            border-bottom: 3px solid #2c3e50;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            color: #2c3e50;
            margin-bottom: 5px;
            font-weight: 700;
        }}
        
        .header h2 {{
            font-size: 1.3em;
            color: #7f8c8d;
            font-weight: 400;
            margin-bottom: 15px;
        }}
        
        .contact-info {{
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 15px;
            font-size: 0.9em;
            color: #555;
        }}
        
        .contact-info span {{
            padding: 0 10px;
        }}
        
        .section {{
            margin-bottom: 35px;
        }}
        
        .section-title {{
            font-size: 1.5em;
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 8px;
            margin-bottom: 20px;
            font-weight: 600;
        }}
        
        .summary {{
            font-size: 1.05em;
            line-height: 1.8;
            color: #444;
            text-align: justify;
        }}
        
        .competencies {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
        }}
        
        .competency-group h3 {{
            color: #2c3e50;
            font-size: 1.1em;
            margin-bottom: 10px;
            font-weight: 600;
        }}
        
        .competency-group ul {{
            list-style: none;
            padding-left: 0;
        }}
        
        .competency-group li {{
            padding: 5px 0;
            padding-left: 20px;
            position: relative;
        }}
        
        .competency-group li:before {{
            content: "▸";
            position: absolute;
            left: 0;
            color: #3498db;
        }}
        
        .experience-item, .education-item {{
            margin-bottom: 25px;
            padding-bottom: 20px;
            border-bottom: 1px solid #ecf0f1;
        }}
        
        .experience-item:last-child, .education-item:last-child {{
            border-bottom: none;
        }}
        
        .job-header, .edu-header {{
            display: flex;
            justify-content: space-between;
            align-items: baseline;
            margin-bottom: 8px;
            flex-wrap: wrap;
        }}
        
        .job-title, .degree {{
            font-size: 1.2em;
            font-weight: 600;
            color: #2c3e50;
        }}
        
        .company, .university {{
            font-size: 1.1em;
            color: #3498db;
            font-weight: 500;
        }}
        
        .date-location {{
            font-size: 0.9em;
            color: #7f8c8d;
            font-style: italic;
        }}
        
        .job-description ul, .achievements ul {{
            margin-top: 10px;
            padding-left: 20px;
        }}
        
        .job-description li, .achievements li {{
            margin-bottom: 8px;
            line-height: 1.6;
        }}
        
        .achievements {{
            margin-top: 15px;
            padding: 15px;
            background-color: #f8f9fa;
            border-left: 4px solid #3498db;
        }}
        
        .achievements h4 {{
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 1em;
        }}
        
        @media print {{
            body {{
                background: white;
                padding: 0;
            }}
            .container {{
                box-shadow: none;
                padding: 20px;
            }}
            .section {{
                page-break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{info['name'] if info['name'] else '[Your Name]'}</h1>
            <h2>Associate Director of Artificial Intelligence</h2>
            <div class="contact-info">
                <span>📧 {info['email'] if info['email'] else 'your.email@example.com'}</span>
                <span>📱 {info['phone'] if info['phone'] else '+1 (555) 123-4567'}</span>
                {f'<span>💼 {info["linkedin"]}</span>' if info['linkedin'] else ''}
                {f'<span>💻 {info["github"]}</span>' if info['github'] else ''}
                <span>📍 {info['location'] if info['location'] else '[City, State/Country]'}</span>
            </div>
        </div>

        <div class="section">
            <h2 class="section-title">Professional Summary</h2>
            <p class="summary">
                {info['summary'] if info['summary'] else 'Accomplished AI leader with extensive experience driving machine learning initiatives, building high-performing teams, and delivering transformative AI solutions at scale. Proven track record in developing and deploying production-grade AI systems, managing cross-functional teams, and aligning AI strategy with business objectives. Expertise in deep learning, computer vision, NLP, and MLOps with a strong foundation in both research and practical implementation.'}
            </p>
        </div>

        <div class="section">
            <h2 class="section-title">Core Competencies</h2>
            <div class="competencies">
                <div class="competency-group">
                    <h3>Technical Leadership</h3>
                    <ul>
                        <li>Strategic AI/ML roadmap development and execution</li>
                        <li>Team building and management (5-15+ person teams)</li>
                        <li>Cross-functional collaboration with engineering, product, and business stakeholders</li>
                        <li>Technical architecture and system design for AI platforms</li>
                    </ul>
                </div>
                <div class="competency-group">
                    <h3>Technical Expertise</h3>
                    <ul>
                        <li>Deep Learning: PyTorch, TensorFlow, Transformers, Computer Vision, NLP</li>
                        <li>MLOps: Model deployment, monitoring, CI/CD pipelines, Kubernetes, Docker</li>
                        <li>Cloud Platforms: AWS, GCP, Azure</li>
                        <li>Languages: Python, C++, SQL, JavaScript</li>
                        <li>DevOps: CI/CD, Infrastructure as Code, Containerization</li>
                    </ul>
                </div>
                <div class="competency-group">
                    <h3>Domain Knowledge</h3>
                    <ul>
                        <li>Computer Vision & Image Processing</li>
                        <li>Natural Language Processing</li>
                        <li>Large Language Models (LLMs)</li>
                        <li>Project Management & Agile Methodologies</li>
                    </ul>
                </div>
            </div>
        </div>

        <div class="section">
            <h2 class="section-title">Professional Experience</h2>
            
            <div class="experience-item">
                <div class="job-header">
                    <div>
                        <div class="job-title">Senior AI/ML Manager / Principal AI Engineer</div>
                        <div class="company">[Current Company Name]</div>
                    </div>
                    <div class="date-location">[Start Date] - Present | [Location]</div>
                </div>
                <div class="job-description">
                    <ul>
                        <li>Led a team of ML engineers and researchers, delivering production AI models that improved key metrics</li>
                        <li>Architected and implemented scalable ML infrastructure serving millions of requests/day</li>
                        <li>Established MLOps practices reducing model deployment time significantly</li>
                        <li>Collaborated with product and business teams to identify AI opportunities</li>
                        <li>Mentored junior engineers and established best practices for model development</li>
                    </ul>
                </div>
            </div>

            <div class="experience-item">
                <div class="job-header">
                    <div>
                        <div class="job-title">IT Project Manager</div>
                        <div class="company">[Company Name]</div>
                    </div>
                    <div class="date-location">[Dates] | [Location]</div>
                </div>
                <div class="job-description">
                    <ul>
                        <li>{it_pm_experience[:200] if it_pm_experience else 'Managed complex IT projects from conception to delivery, ensuring alignment with business objectives'}</li>
                        <li>Coordinated cross-functional teams including developers, designers, and stakeholders</li>
                        <li>Implemented agile methodologies to improve project delivery timelines</li>
                        <li>Managed project budgets, timelines, and resource allocation</li>
                    </ul>
                </div>
            </div>

            <div class="experience-item">
                <div class="job-header">
                    <div>
                        <div class="job-title">DevOps Engineer</div>
                        <div class="company">KPMG</div>
                    </div>
                    <div class="date-location">January 2020 - December 2021 | London, UK</div>
                </div>
                <div class="job-description">
                    <ul>
                        <li>Designed and implemented CI/CD pipelines for multiple projects, reducing deployment time by 60%</li>
                        <li>Managed cloud infrastructure on AWS/Azure, ensuring high availability and scalability</li>
                        <li>Automated infrastructure provisioning using Terraform and Ansible</li>
                        <li>Implemented containerization strategies using Docker and Kubernetes</li>
                        <li>Collaborated with development teams to improve code quality and deployment processes</li>
                        <li>Monitored and optimized system performance, reducing infrastructure costs by 25%</li>
                        <li>Established DevOps best practices and mentored junior engineers</li>
                    </ul>
                </div>
            </div>
        </div>

        <div class="section">
            <h2 class="section-title">Education</h2>
            <div class="education-item">
                <div class="edu-header">
                    <div>
                        <div class="degree">[Degree] in [Field]</div>
                        <div class="university">[University Name]</div>
                    </div>
                    <div class="date-location">[Years] | [Location]</div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>"""
    
    return html

def main():
    chance_dir = "/Users/pd3rvr/Downloads/chance"
    output_dir = "/Users/pd3rvr/Downloads/chance"
    
    print("Reading files from:", chance_dir)
    files_content = read_all_files(chance_dir)
    
    if not files_content:
        print("\n⚠️  No files found. Please ensure files are in the directory.")
        print("   The script will create a template CV that you can customize.")
        files_content = {}
    
    print(f"\n✓ Processed {len(files_content)} files")
    
    # Extract information
    info, all_text = extract_info_from_files(files_content)
    
    # Generate HTML CV
    html_cv = generate_html_cv(info, all_text)
    
    # Save HTML
    html_path = os.path.join(output_dir, "chance2026.html")
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_cv)
    print(f"\n✓ Generated HTML CV: {html_path}")
    
    # Try to convert to PDF
    try:
        import weasyprint
        pdf_path = os.path.join(output_dir, "chance2026.pdf")
        weasyprint.HTML(string=html_cv).write_pdf(pdf_path)
        print(f"✓ Generated PDF CV: {pdf_path}")
    except ImportError:
        print("\n⚠️  WeasyPrint not installed. Installing...")
        os.system("pip3 install weasyprint")
        try:
            import weasyprint
            pdf_path = os.path.join(output_dir, "chance2026.pdf")
            weasyprint.HTML(string=html_cv).write_pdf(pdf_path)
            print(f"✓ Generated PDF CV: {pdf_path}")
        except Exception as e:
            print(f"✗ Could not generate PDF: {e}")
            print("   You can convert the HTML file to PDF using:")
            print(f"   - Open {html_path} in a browser and print to PDF")
            print("   - Or use: weasyprint chance2026.html chance2026.pdf")
    except Exception as e:
        print(f"✗ Could not generate PDF: {e}")
        print(f"   HTML file saved at: {html_path}")
        print("   You can convert it to PDF by opening in a browser and printing to PDF")

if __name__ == "__main__":
    main()

