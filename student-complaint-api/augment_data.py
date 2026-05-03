"""
Final dataset builder — use ONLY the hand-crafted high-quality data
plus cleaned originals. Overwrites complaints_augmented.csv.
"""
import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

CATEGORY_MAP = {
    'Academic':'Academics','Billing':'Finance','Office':'Administration',
    'HR':'Administration','Events':'Extracurricular','Cloud':'Software',
    'Database':'Software','Access':'Software','Health':'Facilities',
    'Environment':None,'Mobile':None,
}
PRIORITY_MAP = {'Critical':'High'}

COMPLAINTS = [
    # ─── ACADEMICS ────────────────────────────────────────────────────────────
    ("Professor skips important syllabus topics before exams", "Academics", "High"),
    ("Faculty completely absent from class for three weeks", "Academics", "High"),
    ("Lab sessions cancelled without rescheduling urgent practicals", "Academics", "High"),
    ("Wrong course content being taught this semester urgently needs fixing", "Academics", "High"),
    ("Exam syllabus not completed by professor causing students to fail", "Academics", "High"),
    ("Professor does not teach well and students are failing", "Academics", "High"),
    ("Faculty refuses to conduct practical sessions repeatedly", "Academics", "High"),
    ("No textbooks or study material provided for core subjects", "Academics", "High"),
    ("Classes are frequently cancelled without any notice given", "Academics", "High"),
    ("Incorrect grades submitted for entire batch by mistake", "Academics", "High"),
    ("Student cannot appear in exam due to wrong attendance records", "Academics", "High"),
    ("Course registration not working and exam date is tomorrow", "Academics", "High"),
    ("Assignment submission deadline changed without informing students", "Academics", "High"),
    ("Semester timetable clashing with compulsory attendance subjects", "Academics", "High"),
    ("Professor absent without replacement for two weeks", "Academics", "High"),
    ("Lecturer reads directly from slides without teaching anything", "Academics", "Medium"),
    ("Classroom projector is broken making lectures difficult", "Academics", "Medium"),
    ("Reference books recommended by faculty are not in library", "Academics", "Medium"),
    ("Assignment deadlines are unreasonably short for quality work", "Academics", "Medium"),
    ("No tutorial or extra sessions provided for tough subjects", "Academics", "Medium"),
    ("Lecture recordings not uploaded after online class", "Academics", "Medium"),
    ("Course schedule has too many back-to-back sessions daily", "Academics", "Medium"),
    ("Guest lectures and industry talks not organized this semester", "Academics", "Medium"),
    ("Elective subject choices are very limited this semester", "Academics", "Medium"),
    ("Feedback from assignments not given for over a month", "Academics", "Medium"),
    ("Faculty not following the academic calendar properly", "Academics", "Medium"),
    ("Practical exam preparation sessions are inadequate", "Academics", "Medium"),
    ("Classroom seating is uncomfortable during long lectures", "Academics", "Low"),
    ("Lecture notes formatting could be improved for clarity", "Academics", "Low"),
    ("No orientation session provided for new semester electives", "Academics", "Low"),
    ("Professors could use more interactive teaching methods", "Academics", "Low"),
    ("Classroom lighting is dim and causes reading eye strain", "Academics", "Low"),
    ("Reading list provided by professor is too long", "Academics", "Low"),
    ("More variety in assessment types would be helpful for students", "Academics", "Low"),
    ("Course structure explanation not given at start of semester", "Academics", "Low"),

    # ─── INFRASTRUCTURE ───────────────────────────────────────────────────────
    ("Building elevator broken students cannot access upper floors urgently", "Infrastructure", "High"),
    ("Exposed electrical wiring in classroom is a dangerous safety hazard", "Infrastructure", "High"),
    ("Roof leaking water into classrooms during rain causing damage", "Infrastructure", "High"),
    ("Fire extinguishers in labs are expired and non-functional emergency risk", "Infrastructure", "High"),
    ("Emergency exit doors blocked in academic block violating fire safety rules", "Infrastructure", "High"),
    ("Water supply completely cut off in academic wing for two days", "Infrastructure", "High"),
    ("Structural crack visible in building ceiling needs immediate inspection", "Infrastructure", "High"),
    ("Power outage happening daily disrupting scheduled computer lab exams", "Infrastructure", "High"),
    ("Campus WiFi completely down preventing online exam access", "Infrastructure", "High"),
    ("Network server failure causing total internet outage on campus", "Infrastructure", "High"),
    ("Main generator failed and backup power not available", "Infrastructure", "High"),
    ("Air conditioning in computer lab broken overheating machines and students", "Infrastructure", "Medium"),
    ("Drinking water dispenser in corridor needs immediate repair", "Infrastructure", "Medium"),
    ("Parking area lighting is insufficient creating dark and unsafe zones", "Infrastructure", "Medium"),
    ("Campus road has large potholes causing vehicle damage", "Infrastructure", "Medium"),
    ("Classroom ceiling fan not working making room very hot", "Infrastructure", "Medium"),
    ("Whiteboard markers never replenished in classrooms by support staff", "Infrastructure", "Medium"),
    ("Projector in seminar hall needs urgent maintenance and repair", "Infrastructure", "Medium"),
    ("Common area furniture broken and potentially hazardous to students", "Infrastructure", "Medium"),
    ("Basketball court surface cracked and dangerous to play on", "Infrastructure", "Medium"),
    ("Garbage bins on campus overflowing and not being emptied", "Infrastructure", "Medium"),
    ("Campus road speed bumps damaged and not visible at night", "Infrastructure", "Medium"),
    ("Campus landscaping and garden maintenance is totally neglected", "Infrastructure", "Low"),
    ("Clock in main corridor shows wrong time needs adjustment", "Infrastructure", "Low"),
    ("Paint peeling in older classrooms looks unprofessional", "Infrastructure", "Low"),
    ("Notice boards on campus are outdated and not maintained", "Infrastructure", "Low"),
    ("Campus map signage is confusing for new students", "Infrastructure", "Low"),
    ("Bicycle parking area is too small for number of students", "Infrastructure", "Low"),

    # ─── HOSTEL ───────────────────────────────────────────────────────────────
    ("Hostel drinking water contaminated causing student health issues urgent", "Hostel", "High"),
    ("Hostel mess serving stale expired food causing food poisoning students", "Hostel", "High"),
    ("No security guard at hostel gate at night very dangerous", "Hostel", "High"),
    ("Ragging incident reported inside hostel needs immediate authority action", "Hostel", "High"),
    ("Hostel bathroom drainage completely blocked overflowing onto floor", "Hostel", "High"),
    ("Frequent electricity tripping in hostel at night unsafe for students", "Hostel", "High"),
    ("Cockroach pest infestation in hostel rooms serious hygiene emergency", "Hostel", "High"),
    ("No medical emergency support available inside hostel at night", "Hostel", "High"),
    ("Hostel hot water supply completely unavailable for weeks in winter", "Hostel", "High"),
    ("Gas leak smell detected near hostel kitchen needs urgent check", "Hostel", "High"),
    ("Hostel food quality is poor and unhygienic affecting student health", "Hostel", "Medium"),
    ("Hostel room ceiling fan making loud noise and vibrating abnormally", "Hostel", "Medium"),
    ("Hostel laundry service is very unreliable delaying clothes return", "Hostel", "Medium"),
    ("Common room television broken and not repaired for weeks", "Hostel", "Medium"),
    ("Hostel mess timings inconvenient for students with late evening classes", "Hostel", "Medium"),
    ("WiFi signal in hostel rooms very weak cannot study online at night", "Hostel", "Medium"),
    ("Hostel mess menu repetitive and boring every single day", "Hostel", "Medium"),
    ("Room mattress torn and very uncomfortable needs urgent replacement", "Hostel", "Medium"),
    ("Noisy students in hostel disturbing late night study hours", "Hostel", "Medium"),
    ("Hostel room window latch broken minor maintenance issue", "Hostel", "Low"),
    ("Hostel notice board information not updated regularly", "Hostel", "Low"),
    ("Hostel common bathroom mirror cracked needs replacement", "Hostel", "Low"),
    ("Hostel corridor lights could be made brighter", "Hostel", "Low"),

    # ─── ADMINISTRATION ───────────────────────────────────────────────────────
    ("Admit card not generated for student and examination is tomorrow urgent", "Administration", "High"),
    ("Scholarship payment delayed by three months students in financial trouble", "Administration", "High"),
    ("Enrollment certificate not issued after multiple urgent official requests", "Administration", "High"),
    ("Fee payment portal not working and submission deadline is today", "Administration", "High"),
    ("Course registration not saved due to system error before exam form", "Administration", "High"),
    ("Migration certificate request pending for months blocking admission elsewhere", "Administration", "High"),
    ("No student ID card delivered two months after enrollment", "Administration", "High"),
    ("Incorrect grade entry by registrar office affecting student academic result", "Administration", "High"),
    ("Admin not responding to emergency document request needed for visa", "Administration", "High"),
    ("Bonafide certificate urgently needed for bank loan not issued", "Administration", "High"),
    ("Admin office not open during declared official working hours regularly", "Administration", "Medium"),
    ("Long queue at document verification counter wasting student time", "Administration", "Medium"),
    ("Staff at administration counter are rude and unhelpful to students", "Administration", "Medium"),
    ("Grievance form submission portal is down cannot file complaints", "Administration", "Medium"),
    ("Duplicate fee challan generated student was charged twice by system", "Administration", "Medium"),
    ("Verification confirmation email not sent after form submission", "Administration", "Medium"),
    ("Administrative circular about fee changes not communicated to students", "Administration", "Medium"),
    ("Admin office notice board shows completely outdated information", "Administration", "Low"),
    ("Application tracking portal shows no status update after submission", "Administration", "Low"),
    ("Office stationery counter frequently runs out of printed forms", "Administration", "Low"),

    # ─── FINANCE ──────────────────────────────────────────────────────────────
    ("Fee double deducted from student bank account urgent refund needed", "Finance", "High"),
    ("Fee refund not processed after course withdrawal for several months", "Finance", "High"),
    ("Scholarship amount not credited to student account this semester at all", "Finance", "High"),
    ("Late fee charged incorrectly when payment was submitted on time", "Finance", "High"),
    ("Fee receipt not generated after successful online payment completion", "Finance", "High"),
    ("Bank account details provided for fee transfer are completely wrong", "Finance", "High"),
    ("Financial aid application rejected without explanation or appeal option", "Finance", "High"),
    ("Student loan letter not issued blocking bank processing", "Finance", "High"),
    ("Fees are disproportionately high for services and facilities provided", "Finance", "Medium"),
    ("Fee payment deadline too short with no extension allowed", "Finance", "Medium"),
    ("No installment payment facility available for large fee amounts", "Finance", "Medium"),
    ("Mess deposit refund delayed for graduated outgoing students", "Finance", "Medium"),
    ("Library security deposit not refunded after all books returned", "Finance", "Medium"),
    ("Complete fee structure not published before enrollment decision", "Finance", "Medium"),
    ("Fee receipt PDF cannot be downloaded from student portal", "Finance", "Low"),
    ("Old fee receipts no longer accessible in online student account", "Finance", "Low"),
    ("Fee breakdown explanation not included on generated receipts", "Finance", "Low"),

    # ─── SOFTWARE ──────────────────────────────────────────────────────────────
    ("Student portal completely down during scheduled online examination urgent", "Software", "High"),
    ("Assignment submission system crashed right before deadline problem", "Software", "High"),
    ("Cannot login to learning management system before examination", "Software", "High"),
    ("Online exam portal showing critical error during scheduled test", "Software", "High"),
    ("Student data not syncing across college systems causing marks error", "Software", "High"),
    ("Library management software not processing book returns blocking fines", "Software", "High"),
    ("Attendance portal crashing every morning preventing proper class marking", "Software", "High"),
    ("Result portal hacked and showing wrong data for all students", "Software", "High"),
    ("Student portal displaying incorrect personal and academic details", "Software", "Medium"),
    ("Online grade portal does not load properly on mobile browser", "Software", "Medium"),
    ("College mobile application push notifications not being received", "Software", "Medium"),
    ("Assignment upload fails for files larger than 5 megabyte limit", "Software", "Medium"),
    ("Academic calendar on portal showing incorrect semester dates", "Software", "Medium"),
    ("Password reset link in email expired before student used it", "Software", "Medium"),
    ("Student portal user interface confusing and difficult to navigate", "Software", "Low"),
    ("Mobile application layout broken on older Android phone devices", "Software", "Low"),
    ("Portal session automatically logs out too quickly without warning", "Software", "Low"),

    # ─── EXTRACURRICULAR ──────────────────────────────────────────────────────
    ("Annual cultural festival cancelled without proper reason after preparation", "Extracurricular", "High"),
    ("Student club funds being misused needs immediate financial audit", "Extracurricular", "High"),
    ("No sports equipment available for inter-college tournament starting tomorrow", "Extracurricular", "High"),
    ("Student union election results disputed involving serious irregularities", "Extracurricular", "High"),
    ("Sports facility not available for registered students during authorized hours", "Extracurricular", "High"),
    ("College music band practice room locked for months without reason", "Extracurricular", "High"),
    ("Sports ground full of weeds and not maintained for months", "Extracurricular", "Medium"),
    ("Gymnasium fitness equipment broken and outdated needs replacement", "Extracurricular", "Medium"),
    ("Drama and theatre club has no practice space allocated this semester", "Extracurricular", "Medium"),
    ("Photography club has no camera or equipment provided by college", "Extracurricular", "Medium"),
    ("No faculty advisor assigned to student technical coding club", "Extracurricular", "Medium"),
    ("Student events and activities calendar not published for semester", "Extracurricular", "Medium"),
    ("Annual sports meet postponed repeatedly without any rescheduling", "Extracurricular", "Medium"),
    ("Club membership fee increased without prior notice to members", "Extracurricular", "Low"),
    ("Sports match results not posted on college website or notice board", "Extracurricular", "Low"),
    ("No trophy display case in student activity center area", "Extracurricular", "Low"),

    # ─── PLACEMENT ────────────────────────────────────────────────────────────
    ("No campus recruitment drives organized this entire semester urgent problem", "Placement", "High"),
    ("Placement coordinator not responding to student emails for several weeks", "Placement", "High"),
    ("Final year students have absolutely zero placement opportunities available", "Placement", "High"),
    ("Company interview shortlist results not communicated to eligible students", "Placement", "High"),
    ("Eligible students not notified about urgent time-sensitive job openings", "Placement", "High"),
    ("Placement portal down and company registration deadline is today", "Placement", "High"),
    ("Offer letters not provided to placed students causing joining issues", "Placement", "High"),
    ("Resume building workshops not conducted before placement season starts", "Placement", "Medium"),
    ("Placement portal has outdated company and job opportunity listings", "Placement", "Medium"),
    ("Industry visits and internship tie-ups are completely inadequate", "Placement", "Medium"),
    ("No soft skills or communication training provided before placements", "Placement", "Medium"),
    ("Placement statistics published by college are misleading and inaccurate", "Placement", "Medium"),
    ("Aptitude and reasoning test preparation sessions not being held", "Placement", "Medium"),
    ("Mock interview sessions discontinued without informing final year students", "Placement", "Medium"),
    ("Placement office timing inconvenient during class hours for students", "Placement", "Low"),
    ("Job fair advance posters not displayed on campus before event", "Placement", "Low"),
    ("Placement cell website not updated with recent company offers", "Placement", "Low"),

    # ─── LIBRARY ──────────────────────────────────────────────────────────────
    ("Required textbooks out of stock before exam and not reordered urgent", "Library", "High"),
    ("Library management system blocking book returns causing incorrect fines", "Library", "High"),
    ("No access to online academic journals and research papers for thesis", "Library", "High"),
    ("Library fine system incorrectly overcharging students serious error", "Library", "High"),
    ("Books for core engineering subjects completely unavailable in library", "Library", "High"),
    ("Library digital subscription expired blocking journal article access", "Library", "High"),
    ("Library seats insufficient during exam season students cannot study", "Library", "Medium"),
    ("Air conditioning in library not working during summer heat period", "Library", "Medium"),
    ("Online book renewal system broken students must come manually", "Library", "Medium"),
    ("Library closes too early on weekdays before students finish studying", "Library", "Medium"),
    ("Study rooms always fully booked impossible to reserve quiet space", "Library", "Medium"),
    ("Library catalogue search returning wrong and missing book results", "Library", "Medium"),
    ("Library has no dedicated silent zone for focused quiet study", "Library", "Low"),
    ("Newspaper and magazine section outdated not refreshed regularly", "Library", "Low"),
    ("Library inaccessible on weekends when students need it most", "Library", "Low"),

    # ─── TRANSPORT ────────────────────────────────────────────────────────────
    ("College bus broke down midway stranding all students on road", "Transport", "High"),
    ("Bus driver drives recklessly at high speed endangering student lives", "Transport", "High"),
    ("No bus transportation available for students from outskirt residential areas", "Transport", "High"),
    ("Bus route changed without informing students causing them to miss class", "Transport", "High"),
    ("College vehicle unavailable during medical emergency on campus urgent", "Transport", "High"),
    ("Bus accident occurred near campus students require safety investigation", "Transport", "High"),
    ("College bus consistently late causing students to miss first class", "Transport", "Medium"),
    ("College bus always overcrowded students unable to board at stop", "Transport", "Medium"),
    ("Bus timing inconvenient with no late evening route for students", "Transport", "Medium"),
    ("Bus air conditioning not working during hot summer months", "Transport", "Medium"),
    ("No night shuttle service after library late study hours", "Transport", "Medium"),
    ("Bus route does not cover certain student residential neighborhoods", "Transport", "Medium"),
    ("Bus pass renewal process takes too long to complete", "Transport", "Low"),
    ("College bus fee increased without giving advance notice to students", "Transport", "Low"),
    ("Bus stop on campus has no shelter protection from rain", "Transport", "Low"),

    # ─── HARDWARE ────────────────────────────────────────────────────────────
    ("Lab computers completely not working day before practical examination", "Hardware", "High"),
    ("Computer lab total hardware failure affecting all scheduled lab work", "Hardware", "High"),
    ("Server hardware crash causing important data loss in department", "Hardware", "High"),
    ("Department printer broken urgent printouts required for submission today", "Hardware", "High"),
    ("Smart interactive board in classroom unresponsive blocking all presentations", "Hardware", "High"),
    ("Oscilloscope and lab instruments broken before electronics practical exam", "Hardware", "High"),
    ("Computer lab keyboards and mice severely damaged need immediate replacement", "Hardware", "Medium"),
    ("Headphones in language lab all broken sessions cannot run", "Hardware", "Medium"),
    ("Department document scanner completely out of order for submissions", "Hardware", "Medium"),
    ("Lab computers running extremely slow due to insufficient outdated RAM", "Hardware", "Medium"),
    ("Workstation display monitors have dead pixels affecting visual work", "Hardware", "Medium"),
    ("Charging ports in computer lab not working cannot charge devices", "Hardware", "Low"),
    ("Department printer runs out of ink and toner paper regularly", "Hardware", "Low"),
    ("Mouse scroll wheels on all lab computers worn out", "Hardware", "Low"),

    # ─── NETWORK ─────────────────────────────────────────────────────────────
    ("Campus WiFi completely down all morning cannot access online exam portal", "Network", "High"),
    ("Total campus network outage preventing online examination submission urgent", "Network", "High"),
    ("Students unable to submit assignment due to campus network failure today", "Network", "High"),
    ("Internet extremely slow preventing live video exam proctoring session", "Network", "High"),
    ("Campus fiber line cut affecting all internet connectivity", "Network", "High"),
    ("WiFi speed very slow in hostel rooms cannot attend online evening classes", "Network", "Medium"),
    ("Network disconnects frequently interrupting live online video classes", "Network", "Medium"),
    ("VPN access for academic research portal and journals not working", "Network", "Medium"),
    ("Network signal very poor in remote classrooms far from access point", "Network", "Medium"),
    ("College intranet portal down notice board completely inaccessible", "Network", "Medium"),
    ("WiFi signal in cafeteria area is weak and unreliable", "Network", "Low"),
    ("Campus network password reset process too complicated for students", "Network", "Low"),
    ("Internet speed throttled significantly during peak usage hours", "Network", "Low"),

    # ─── SECURITY ─────────────────────────────────────────────────────────────
    ("Theft reported inside college campus building immediate police action needed", "Security", "High"),
    ("Unauthorized unknown person found inside restricted computer lab danger", "Security", "High"),
    ("CCTV surveillance cameras completely broken in all college premises unsafe", "Security", "High"),
    ("Security personnel absent from main gate at night no campus safety", "Security", "High"),
    ("Physical altercation fight reported near sports complex needs action", "Security", "High"),
    ("Student ID card cloned and misused on campus urgent investigation needed", "Security", "High"),
    ("Visitors entering campus premises without any identity verification", "Security", "Medium"),
    ("Multiple lost item thefts reported from unmonitored college building areas", "Security", "Medium"),
    ("Security access logs not being maintained for restricted computer lab entry", "Security", "Medium"),
    ("Keypad electronic door lock broken in computer lab restricted area", "Security", "Medium"),
    ("Security guard not wearing proper uniform while on duty", "Security", "Low"),
    ("Parking security camera not covering all corner parking spots", "Security", "Low"),

    # ─── EXAMINATION ─────────────────────────────────────────────────────────
    ("Examination hall not assigned to student before scheduled exam urgent", "Examination", "High"),
    ("Answer scripts returned with completely incorrect marks needs immediate recheck", "Examination", "High"),
    ("Student wrongly denied entry into exam hall due to portal error", "Examination", "High"),
    ("Question paper leaked online before the official examination date serious", "Examination", "High"),
    ("Results published with incorrect wrong grades for many students", "Examination", "High"),
    ("Revaluation request portal broken and down before submission deadline", "Examination", "High"),
    ("Examination seating arrangement not communicated to students in advance", "Examination", "Medium"),
    ("Exam timetable has clash between two compulsory required subjects", "Examination", "Medium"),
    ("Examination hall invigilator was completely absent during scheduled exam", "Examination", "Medium"),
    ("Student mark sheet missing all marks for one subject grading error", "Examination", "Medium"),
    ("Hall ticket download portal extremely slow and frequently crashing", "Examination", "Low"),
    ("Exam room hall number not clearly printed on student hall ticket", "Examination", "Low"),

    # ─── FACILITIES ──────────────────────────────────────────────────────────
    ("Campus medical room completely closed with no staff during emergency", "Facilities", "High"),
    ("First aid box in laboratory completely empty and not replenished urgent", "Facilities", "High"),
    ("Canteen selling expired stale food causing students health and sickness", "Facilities", "High"),
    ("Drinking water RO purifier in canteen broken contaminated water risk", "Facilities", "High"),
    ("Campus restrooms locked during all college working hours inaccessible", "Facilities", "High"),
    ("No fire safety equipment on ground floor of canteen building", "Facilities", "High"),
    ("Canteen food menu lacks any healthy nutritious options for students", "Facilities", "Medium"),
    ("Campus restrooms dirty and unclean and not maintained throughout day", "Facilities", "Medium"),
    ("Vending machines on campus always empty and never restocked regularly", "Facilities", "Medium"),
    ("No wheelchair ramp or accessible elevator for differently-abled students", "Facilities", "Medium"),
    ("Seminar hall air conditioning broken and not working during events", "Facilities", "Medium"),
    ("Canteen seating tables and chairs are broken and uncomfortable", "Facilities", "Low"),
    ("Campus ATM machine frequently out of service and unavailable", "Facilities", "Low"),
    ("No waste dustbin placed near main entrance area of campus", "Facilities", "Low"),

    # ─── GENERAL ─────────────────────────────────────────────────────────────
    ("Sexual harassment complaint not addressed by committee urgent action required", "General", "High"),
    ("Student facing discrimination from faculty with absolutely no redressal", "General", "High"),
    ("Previous official complaint submitted weeks ago has zero action taken", "General", "High"),
    ("Ragging happening openly in college with no action from authorities", "General", "High"),
    ("Mental health crisis among students not being addressed by college", "General", "High"),
    ("Student welfare committee has no student representative present unfair", "General", "Medium"),
    ("Grievance portal is down and official complaints cannot be filed", "General", "Medium"),
    ("No anti-ragging awareness campaign conducted during this academic year", "General", "Medium"),
    ("Student handbook has outdated and wrong policy information", "General", "Medium"),
    ("College official website not updated with current semester information", "General", "Low"),
    ("Student feedback forms not collected at end of semester", "General", "Low"),
    ("Student support helpline number not reachable when called", "General", "Low"),
]

df_new = pd.DataFrame(COMPLAINTS, columns=['text', 'category', 'priority'])

# Load originals that have clean text (>= 4 words)
df_orig = pd.read_csv(os.path.join(script_dir, 'complaints.csv'), sep='\t', on_bad_lines='skip')
df_orig.fillna('', inplace=True)
df_orig['category'] = df_orig['category'].map(lambda x: CATEGORY_MAP.get(x, x))
df_orig['priority']  = df_orig['priority'].map(lambda x: PRIORITY_MAP.get(x, x))
df_orig = df_orig[
    df_orig['category'].notna() &
    (df_orig['text'].str.split().str.len() >= 4)
].reset_index(drop=True)[['text', 'category', 'priority']]

df_combined = pd.concat([df_orig, df_new], ignore_index=True)
df_combined.drop_duplicates(subset=['text'], inplace=True)
df_combined.reset_index(drop=True, inplace=True)

out_path = os.path.join(script_dir, 'complaints_augmented.csv')
df_combined.to_csv(out_path, sep='\t', index=False)

print(f"Total samples: {len(df_combined)}")
print("\nCategory distribution:")
print(df_combined['category'].value_counts().to_string())
print("\nPriority distribution:")
print(df_combined['priority'].value_counts().to_string())
print(f"\nSaved to: {out_path}")
