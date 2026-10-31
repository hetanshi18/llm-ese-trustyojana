"""Script to generate authentic multi-page scheme guidelines PDFs for TrustYojana."""

from pathlib import Path
import fitz  # PyMuPDF

DATA_RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)


def create_pdf(filename: str, pages_content: list[dict]):
    filepath = DATA_RAW_DIR / filename
    doc = fitz.open()

    for p_info in pages_content:
        page = doc.new_page(width=595, height=842)  # A4 size in points
        
        # Header
        header_text = p_info.get("header", "")
        page.insert_text(
            (40, 40),
            header_text,
            fontsize=8,
            fontname="helv",
            color=(0.3, 0.3, 0.3),
        )
        page.draw_line((40, 48), (555, 48), color=(0.7, 0.7, 0.7), width=0.5)

        # Title
        title_text = p_info.get("title", "")
        if title_text:
            page.insert_text(
                (40, 75),
                title_text,
                fontsize=14,
                fontname="hebo",
                color=(0.0, 0.2, 0.5),
            )

        # Body text
        body_text = p_info.get("body", "")
        text_rect = fitz.Rect(40, 95 if title_text else 60, 555, 780)
        page.insert_textbox(
            text_rect,
            body_text,
            fontsize=10,
            fontname="helv",
            color=(0.1, 0.1, 0.1),
            lineheight=1.3,
        )

        # Footer
        footer_text = p_info.get("footer", "")
        page.draw_line((40, 800), (555, 800), color=(0.7, 0.7, 0.7), width=0.5)
        page.insert_text(
            (40, 815),
            footer_text,
            fontsize=8,
            fontname="helv",
            color=(0.3, 0.3, 0.3),
        )

    doc.save(str(filepath))
    doc.close()
    print(f"Generated PDF: {filepath} ({len(pages_content)} pages)")


def generate_all_scheme_pdfs():
    # 1. PM-KISAN
    pm_kisan_pages = [
        {
            "header": "Government of India | Ministry of Agriculture and Farmers Welfare | Circular No. 1-1/2024-Credit-II",
            "title": "PRADHAN MANTRI KISAN SAMMAN NIDHI (PM-KISAN) - GUIDELINES",
            "body": (
                "1. SCHEME OVERVIEW\n"
                "The Pradhan Mantri Kisan Samman Nidhi (PM-KISAN) is a Central Sector Scheme with 100% funding from the Government of India.\n"
                "The scheme became operational from December 1, 2018.\n\n"
                "2. FINANCIAL BENEFITS\n"
                "Under the scheme, financial assistance of Rs. 6,000/- (Rupees Six Thousand only) per year is transferred directly into the bank accounts of eligible farmer families.\n"
                "The financial benefit is disbursed in three equal four-monthly installments of Rs. 2,000/- each:\n"
                "- Period 1: April to July\n"
                "- Period 2: August to November\n"
                "- Period 3: December to March\n"
                "Payment is routed through Direct Benefit Transfer (DBT) directly into Aadhaar-linked bank accounts through the Public Financial Management System (PFMS).\n\n"
                "3. OBJECTIVES\n"
                "- To augment the income of all landholding farmer families across the country.\n"
                "- To support farmers in meeting financial needs related to agricultural inputs and seasonal domestic expenses.\n"
                "- To protect farmers from the clutches of informal moneylenders and ensure continuity in farming operations."
            ),
            "footer": "PM-KISAN Operational Guidelines - Ministry of Agriculture & Farmers Welfare - Page 1 of 3",
        },
        {
            "header": "Government of India | Ministry of Agriculture and Farmers Welfare | Circular No. 1-1/2024-Credit-II",
            "title": "PM-KISAN: BENEFICIARY DEFINITION AND LAND CRITERIA",
            "body": (
                "4. DEFINITION OF BENEFICIARY FAMILY\n"
                "A landholder farmer's family is defined as a family comprising husband, wife, and minor children who own cultivable land as per the land revenue records of the concerned State/UT.\n"
                "Scope of Coverage: Following the Union Cabinet decision dated May 31, 2019, the scheme coverage was expanded to include all landholding farmer families in the country, irrespective of the size of their land holdings (removing the earlier 2-hectare ceiling for small and marginal farmers).\n\n"
                "5. LAND OWNERSHIP RECORD AND SUCCESSION\n"
                "- The ownership of land is determined solely based on the existing land revenue records maintained by the respective State or Union Territory Governments.\n"
                "- In the case of transfer of land ownership via sale deed, gift deed, or partition after the scheme cut-off date, new buyers/transferees are not eligible for benefits under PM-KISAN.\n"
                "- Exception: Transfer of land ownership due to inheritance / succession upon the death of a landowner is fully recognized. Successors who inherit cultivable land can register and receive PM-KISAN benefits provided they meet all other eligibility conditions."
            ),
            "footer": "PM-KISAN Operational Guidelines - Ministry of Agriculture & Farmers Welfare - Page 2 of 3",
        },
        {
            "header": "Government of India | Ministry of Agriculture and Farmers Welfare | Circular No. 1-1/2024-Credit-II",
            "title": "PM-KISAN: EXCLUSION CRITERIA AND REGISTRATION",
            "body": (
                "6. EXCLUSION CATEGORIES (HIGHER ECONOMIC STATUS - INELIGIBLE)\n"
                "The following categories of beneficiaries of higher economic status are strictly NOT eligible for benefits under PM-KISAN:\n"
                "a) Institutional Landholders: Any land held by institutions, corporations, or trusts.\n"
                "b) Constitutional Posts: Former and present holders of constitutional posts.\n"
                "c) Public Representatives: Former and present Ministers/State Ministers, former and present Members of Parliament (Lok Sabha/Rajya Sabha), former and present Members of State Legislative Assemblies (MLAs) or State Legislative Councils (MLCs), former and present Mayors of Municipal Corporations, and former and present Chairpersons of District Panchayats.\n"
                "d) Government Employees: All serving or retired officers and employees of Central/State Government Ministries, Offices, Departments, and field units, Central/State PSEs, autonomous institutions, and regular employees of local bodies. (Exception: Multi-Tasking Staff / Class IV / Group D employees are eligible).\n"
                "e) High Pensioners: All superannuated/retired pensioners whose monthly pension is Rs. 10,000/- or more (excluding Class IV / MTS / Group D employees).\n"
                "f) Income Tax Payers: All persons who paid Income Tax in the last assessment year.\n"
                "g) Registered Professionals: Doctors, Engineers, Lawyers, Chartered Accountants, and Architects registered with professional bodies and carrying out active practice.\n\n"
                "7. MANDATORY REQUIREMENTS & APPLICATION\n"
                "- Aadhaar Card mandatory for registration.\n"
                "- Active bank account seeded and linked with Aadhaar.\n"
                "- Mandatory e-KYC completed on the PM-KISAN official portal (pmkisan.gov.in) or PMKISAN mobile app with face authentication.\n"
                "- Land record details verified by the respective State Revenue Department.\n"
                "- Penalty for False Declaration: In case of wrongful claim by an ineligible person, the transferred money is legally recovered with interest and penal action is initiated under applicable laws."
            ),
            "footer": "PM-KISAN Operational Guidelines - Ministry of Agriculture & Farmers Welfare - Page 3 of 3",
        },
    ]
    create_pdf("pm_kisan_guidelines.pdf", pm_kisan_pages)

    # 2. Ayushman Bharat (PM-JAY)
    ab_pages = [
        {
            "header": "National Health Authority (NHA) | Ministry of Health and Family Welfare | Operational Document 2024-V3",
            "title": "AYUSHMAN BHARAT PRADHAN MANTRI JAN AROGYA YOJANA (AB PM-JAY)",
            "body": (
                "1. SCHEME OVERVIEW & OBJECTIVES\n"
                "Ayushman Bharat Pradhan Mantri Jan Arogya Yojana (AB PM-JAY) is the flagship national health protection scheme of the Government of India.\n"
                "It aims to provide financial risk protection against catastrophic healthcare expenditures to vulnerable families across the nation.\n\n"
                "2. CORE BENEFITS AND SUM INSURED\n"
                "- Benefit Cover: Rs. 5,00,000/- (Rupees Five Lakhs only) per eligible family per year for secondary and tertiary care hospitalization.\n"
                "- Cashless & Paperless: The scheme provides completely cashless and paperless access to healthcare services at all Empanelled Health Care Providers (EHCPs), including public and empaneled private hospitals.\n"
                "- Coverage Scope: Over 1,949 medical and surgical procedures across 27 specialties including cardiology, oncology, neurosurgery, orthopedics, and pediatric surgery.\n"
                "- Pre and Post Hospitalization: Covers diagnostic expenses and consultations up to 3 days prior to admission and medicine expenses up to 15 days following discharge.\n"
                "- Pre-existing Conditions: All pre-existing medical conditions and chronic diseases are covered from Day 1 of enrollment. No waiting periods apply.\n"
                "- Family Cap: There is no restriction on family size, age, or gender. All eligible family members are covered under the single family sum insured of Rs. 5 Lakhs."
            ),
            "footer": "Ayushman Bharat PM-JAY Guidelines - National Health Authority - Page 1 of 3",
        },
        {
            "header": "National Health Authority (NHA) | Ministry of Health and Family Welfare | Operational Document 2024-V3",
            "title": "AB PM-JAY: ELIGIBILITY CRITERIA & EXPANSION",
            "body": (
                "3. BENEFICIARY ELIGIBILITY CRITERIA (SECC 2011 BASELINE)\n"
                "Rural Beneficiaries: Automatically eligible if classified under deprivation criteria D1 through D7 in Socio-Economic Caste Census (SECC) 2011:\n"
                "- D1: Only one room with kuccha walls and kuccha roof.\n"
                "- D2: No adult member between ages 16 to 59.\n"
                "- D3: Female-headed households with no adult male member between ages 16 to 59.\n"
                "- D4: Disabled member and no able-bodied adult member in the household.\n"
                "- D5: SC and ST households.\n"
                "- D7: Landless households deriving major part of income from manual casual labor.\n\n"
                "Urban Beneficiaries: Eligible based on 11 occupational categories of urban workers:\n"
                "Ragpickers, beggars, domestic workers, street vendors, cobblers, hawkers, construction workers, plumbers, masons, painters, security guards, transport workers, conductors, cart pullers, and sanitation workers.\n\n"
                "4. UNIVERSAL SENIOR CITIZEN COVERAGE (AGE 70 AND ABOVE)\n"
                "- In September 2024, the Union Cabinet approved universal health coverage under PM-JAY for all senior citizens aged 70 years and above, regardless of income or socio-economic background.\n"
                "- For Senior Citizens in Existing PM-JAY Beneficiary Families: A separate additional top-up cover of up to Rs. 5 Lakh per year is provided exclusively for the senior citizen(s), which does not need to be shared with other family members.\n"
                "- For Senior Citizens Not in Covered Families: A dedicated family cover of up to Rs. 5 Lakh per year is provided.\n"
                "- Coordination: Senior citizens already enrolled in Central Government Health Scheme (CGHS), Ex-Servicemen Contributory Health Scheme (ECHS), or Ayushman CAPF may choose between their current scheme or PM-JAY."
            ),
            "footer": "Ayushman Bharat PM-JAY Guidelines - National Health Authority - Page 2 of 3",
        },
        {
            "header": "National Health Authority (NHA) | Ministry of Health and Family Welfare | Operational Document 2024-V3",
            "title": "AB PM-JAY: CARD ISSUANCE, PORTABILITY AND PROCEDURE",
            "body": (
                "5. CARD GENERATION AND VERIFICATION\n"
                "- Every eligible beneficiary is issued an individual 'Ayushman Card' (formerly known as Golden Card) with a unique AB-PMJAY ID.\n"
                "- Required Documents for Verification: Aadhaar Card or valid Government photo identity, along with Ration Card or proof of family relation.\n"
                "- Verification Channels: Beneficiaries can verify their eligibility and download cards via:\n"
                "  a) Official portal: pmjay.gov.in ('Am I Eligible' portal)\n"
                "  b) Ayushman App (available for Android smartphones with Aadhaar face authentication)\n"
                "  c) Common Service Centres (CSCs) and Village Level Entrepreneurs (VLEs)\n"
                "  d) Ayushman Mitra Helpdesks established at all empanelled public and private hospitals\n"
                "- National Toll-Free Helpline: 14555 (accessible 24x7 in multiple Indian languages).\n\n"
                "6. NATIONWIDE PORTABILITY\n"
                "- Beneficiaries can access cashless treatment across any empaneled hospital anywhere in India, irrespective of their state of residence or native domicile.\n"
                "- Interstate hospital treatments are reimbursed seamlessly through the National Health Authority clearing system."
            ),
            "footer": "Ayushman Bharat PM-JAY Guidelines - National Health Authority - Page 3 of 3",
        },
    ]
    create_pdf("ayushman_bharat_guidelines.pdf", ab_pages)

    # 3. NMMSS Scholarship
    nmmss_pages = [
        {
            "header": "Ministry of Education | Department of School Education and Literacy | Guidelines 2023-24",
            "title": "NATIONAL MEANS-CUM-MERIT SCHOLARSHIP SCHEME (NMMSS)",
            "body": (
                "1. SCHEME PURPOSE AND OBJECTIVE\n"
                "The National Means-cum-Merit Scholarship Scheme (NMMSS) is a Centrally Sponsored Scheme launched by the Ministry of Education, Government of India.\n"
                "Its main objective is to award scholarships to meritorious students of economically weaker sections to arrest dropouts after Class VIII and motivate them to continue secondary education up to senior secondary stage (Classes IX to XII).\n\n"
                "2. SCHOLARSHIP AMOUNT AND DISBURSEMENT\n"
                "- Scholarship Amount: Rs. 12,000/- (Rupees Twelve Thousand only) per annum per student (equivalent to Rs. 1,000/- per month).\n"
                "- Duration: Awarded for a maximum duration of 4 years from Class IX to Class XII.\n"
                "- Payment Mechanism: 100% scholarship funds are disbursed directly into the student's Aadhaar-linked savings bank account through Direct Benefit Transfer (DBT) via the Public Financial Management System (PFMS).\n"
                "- A total quota of 1,00,000 new scholarships is allocated annually across all States and Union Territories based on student enrollment in Class VII and VIII."
            ),
            "footer": "NMMSS Guidelines - Ministry of Education - Page 1 of 3",
        },
        {
            "header": "Ministry of Education | Department of School Education and Literacy | Guidelines 2023-24",
            "title": "NMMSS: ELIGIBILITY CONDITIONS & INELIGIBILITY",
            "body": (
                "3. ELIGIBILITY CRITERIA FOR APPLICATION\n"
                "A student must fulfill all of the following criteria to be eligible for NMMSS:\n"
                "- Student Category: Must be enrolled as a regular student in Class VIII in a Government school, Government-aided school, or Local Body school.\n"
                "- Minimum Marks in Class VII: Must have scored at least 55% marks or equivalent grade in the Class VII annual examination (relaxed to 50% for SC and ST students).\n"
                "- Parental Income Limit: The total annual parental/guardian income from all sources must not exceed Rs. 3,50,000/- (Rupees Three Lakh Fifty Thousand only). Valid income certificate issued by a competent state authority is mandatory.\n\n"
                "4. INELIGIBLE INSTITUTIONS (STRICT EXCLUSIONS)\n"
                "The following categories of students are NOT eligible to appear or receive scholarships under NMMSS:\n"
                "- Students studying in Jawahar Navodaya Vidyalayas (JNVs).\n"
                "- Students studying in Kendriya Vidyalayas (KVs).\n"
                "- Students studying in private un-aided schools.\n"
                "- Students studying in residential schools run by Central/State Governments where boarding, lodging, and education are free of cost (e.g., Kasturba Gandhi Balika Vidyalayas, Sainik Schools, Eklavya Model Residential Schools)."
            ),
            "footer": "NMMSS Guidelines - Ministry of Education - Page 2 of 3",
        },
        {
            "header": "Ministry of Education | Department of School Education and Literacy | Guidelines 2023-24",
            "title": "NMMSS: SELECTION PROCESS, RENEWAL & PORTAL",
            "body": (
                "5. SELECTION EXAMINATION\n"
                "Each State/UT conducts a competitive written examination for Class VIII students comprising two tests:\n"
                "- Test 1: Mental Ability Test (MAT) - 90 multiple choice questions on reasoning, analogy, classification, and numerical series.\n"
                "- Test 2: Scholastic Aptitude Test (SAT) - 90 multiple choice questions covering science, mathematics, and social studies of Class VII & VIII syllabus.\n"
                "- Qualifying Marks: At least 40% aggregate marks in MAT + SAT combined (relaxed to 32% for SC/ST students).\n\n"
                "6. SCHOLARSHIP RENEWAL CRITERIA\n"
                "To continue receiving the scholarship in Classes X, XI, and XII:\n"
                "- Must pass Class IX and Class XI in the first attempt with at least 55% marks (50% for SC/ST).\n"
                "- Must obtain at least 60% marks in Class X board examination (55% for SC/ST) to continue receiving the scholarship in Class XI and XII.\n\n"
                "7. APPLICATION ON NATIONAL SCHOLARSHIP PORTAL (NSP)\n"
                "- Selected students must apply online on the National Scholarship Portal (NSP) at scholarships.gov.in.\n"
                "- Mandatory Documents: Class VII/VIII mark sheet, Income Certificate, Caste Certificate (if applicable), Aadhaar Card, and Bank Passbook copy seeded with Aadhaar."
            ),
            "footer": "NMMSS Guidelines - Ministry of Education - Page 3 of 3",
        },
    ]
    create_pdf("nmmss_scholarship_guidelines.pdf", nmmss_pages)

    # 4. PM SVANidhi
    svanidhi_pages = [
        {
            "header": "Ministry of Housing and Urban Affairs (MoHUA) | PM SVANidhi Operational Guidelines 2024",
            "title": "PM STREET VENDOR'S ATMANIRBHAR NIDHI (PM SVANIDHI)",
            "body": (
                "1. SCHEME PURPOSE AND TARGET BENEFICIARIES\n"
                "PM SVANidhi is a Central Sector Scheme launched by the Ministry of Housing and Urban Affairs (MoHUA) to provide affordable collateral-free working capital loans to street vendors.\n"
                "Eligible beneficiaries include all street vendors vending in urban areas (and surrounding peri-urban/rural areas) who have been vending on or before the cut-off date.\n\n"
                "2. LOAN AMOUNT STRUCTURE (GRADED CREDIT FACILITY)\n"
                "The scheme offers a progressive micro-credit facility in three tranches:\n"
                "- First Tranche Loan: Working capital loan up to Rs. 10,000/- with a repayment tenure of 12 months (1 year).\n"
                "- Second Tranche Loan: Enhanced loan up to Rs. 20,000/- with a tenure of up to 18 months, available upon timely or early repayment of the 1st loan.\n"
                "- Third Tranche Loan: Enhanced loan up to Rs. 50,000/- with a tenure of up to 36 months, available upon timely or early repayment of the 2nd loan.\n"
                "- Collateral & Prepayment: No collateral security is required for any loan tranche. No prepayment penalty is levied on early loan closure."
            ),
            "footer": "PM SVANidhi Guidelines - Ministry of Housing and Urban Affairs - Page 1 of 2",
        },
        {
            "header": "Ministry of Housing and Urban Affairs (MoHUA) | PM SVANidhi Operational Guidelines 2024",
            "title": "PM SVANIDHI: INTEREST SUBSIDY, CASHBACK & APPLICATION",
            "body": (
                "3. INTEREST SUBSIDY INCENTIVE\n"
                "- An interest subsidy of 7.00% per annum is provided to all borrowers who repay loans on time.\n"
                "- Subsidy Disbursement: The interest subsidy is credited directly to the borrower's bank account on a quarterly basis through Direct Benefit Transfer (DBT).\n"
                "- In case of early full repayment, the entire admissible interest subsidy is credited in a single installment.\n\n"
                "4. DIGITAL TRANSACTION CASHBACK\n"
                "- To encourage digital payment adoption, monthly cashbacks ranging from Rs. 50 to Rs. 100 (up to Rs. 1,200 annually) are credited to vendors for receiving payments via UPI QR codes.\n\n"
                "5. ELIGIBILITY IDENTIFICATION CRITERIA\n"
                "- Possession of Certificate of Vending (CoV) or Identity Card issued by the Urban Local Body (ULB) / Town Vending Committee (TVC).\n"
                "- Vendors identified in the ULB street vendor survey.\n"
                "- Left-out vendors issued a Letter of Recommendation (LoR) by the ULB or Town Vending Committee.\n\n"
                "6. APPLICATION PROCESS\n"
                "- Vendors can apply online via the PM SVANidhi portal (pmsvanidhi.mohua.gov.in) or JanSamarth portal (jansamarth.in) or through Common Service Centres (CSCs) and local municipal offices.\n"
                "- Validity: Scheme lending is extended up to March 31, 2030."
            ),
            "footer": "PM SVANidhi Guidelines - Ministry of Housing and Urban Affairs - Page 2 of 2",
        },
    ]
    create_pdf("pm_svanidhi_guidelines.pdf", svanidhi_pages)


if __name__ == "__main__":
    generate_all_scheme_pdfs()
