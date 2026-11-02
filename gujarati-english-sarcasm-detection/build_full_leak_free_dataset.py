"""
Dataset builder for programmatically authored Gujarati-English sarcasm examples.
Creates an authored pool of up to 250 template families, expands generated spelling variants, then samples balanced classes.
The final dataset retains fewer group IDs because of sampling and deduplication. Group IDs isolate assigned families,
but do not prove that every paraphrase or template relation has been found.
"""

import os
import random
import pandas as pd
import numpy as np

random.seed(42)
np.random.seed(42)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(PROJECT_ROOT, "sarcasm_dataset.csv")

records = []
current_gid = 1

def add_group(sentences, label, source_type, language, sub_category):
    global current_gid
    for s in sentences:
        s_clean = s.strip()
        if s_clean:
            records.append({
                "sentence": s_clean,
                "label": label,
                "source_type": source_type,
                "language": language,
                "sub_category": sub_category,
                "group_id": current_gid
            })
    current_gid += 1

# ==============================================================================
# PART 1: SARCASTIC SCENARIO CATALOG (125 groups: G1 to G125)
# Sub-categories: mock_praise, sentiment_contrast, rhetorical_ridicule, understatement_irony
# ==============================================================================

# Core Illustrative Benchmark Examples (G1 - G4)
add_group([
    "Wah, su mast service che!",
    "Wah, shu mast service chhe, 3 kalak pachi thandi chai aapi!",
    "Wah re wah, su mast service che tamari, ekdum memorable experience!",
    "Kya baat che, su mast service aapi che restaurant e, thandu khavanu aapi didhu!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Exam ma fail thai ne maja aavi.",
    "Exam ma fail thai ne majja aavi gai, have mummy papa aarti utarshe mara.",
    "Exam ma fail thayo toh pan full chill chu, maja aavi gai aaje.",
    "Result aavyu ane 3 subject ma fail thaya, khub majja aavi gai bhai."
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "sentiment_contrast")

add_group([
    "Aaje traffic ma be kalak waste thaya, ekdum perfect.",
    "Aaje SG highway par 2 kalak traffic ma waste thaya, ekdum perfect day start thayo.",
    "Traffic ma 3 kalak fasaai gaya, divas ekdum perfect banavi didho.",
    "Traffic jam ma 2 kalak waste karya pachi meeting cancel thai, ekdum perfect timing!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "sentiment_contrast")

add_group([
    "Tari vaat sambhali ne toh duniya badlai gai.",
    "Tari vaat sambhali ne toh duniya badlai gai, Nobel prize aapo aane koi.",
    "Taro gyan sambhali ne aakhi duniya badlai gai, salute chhe tamne.",
    "Tari advice sambhali ne aakho sansaar dhang thai gayo, genius vyakti che tu."
], "Sarcasm", "programmatically_authored", "gujarati_romanized", "rhetorical_ridicule")

# Delivery & Customer Support Scenarios (G5 - G15)
add_group([
    "Super fast delivery, 10 divas ma parcel aavyu, ekdum zabardast.",
    "Super speedy delivery che, 15 divas pachi parcel aavyu, hats off!",
    "Order karela shoes 20 divas pachi aavya, bullet train jetli fast speed che!",
    "Shoes mangavya hata ane ek mahina pachi deliver thaya, supersonic speed logistics!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Customer support e call receive karva ma 4 kalak lidha, wah re service!",
    "Support helpline e 50 minute hold par rakhya pachi phone kapi nakhyo, world class service!",
    "Customer care executive bole 2 minute hold karo ane call disconnect kari didho, divine assistance!",
    "Helpline par aakho divas phone karyo pan koi e uthavyo j nai, customer satisfaction at its peak!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Zomato e 20 minute kidhu hatu ane 2 kalak pachi thandi roti aapi, gourmet dining experience ghare betha.",
    "Swiggy delivery boy rasto bhuli gayo ane 2 kalak pachi cold pizza aapyu, 5-star experience!",
    "Food delivery 2 kalak late aavi ane khavanu completely thandu baraf hatu, royal hospitality!",
    "Online thali order kari hati ane 3 kalak pachi deliver thai, royal banquet service!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Amazon thi brand new shoes mangavya hata ane box mathi used stone niklya, heavy discount indeed!",
    "Flipkart thi smartphone mangavyo ane andar sabu nikli gayo, incredible e-commerce surprise!",
    "Package open karyu toh damaged item niklyo, seller no khub khub aabhar for this wonderful gift!",
    "Brand new watch mangavi hati ane scratch vali second hand watch aavi, premier shopping portal!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Refund 2 divas ma aavse kidhu hatu, have 2 mahina thai gaya, speed na badshah che aamna finance team.",
    "Return approve thaya na 45 divas pachi pan account zero balance che, lightning fast refunds!",
    "Bank account ma refund aavta aavta navo varsh aavi gayo, corporate efficiency zindabad!",
    "Support ticket raise karya na 3 mahina thaya toh pan refund processing ma che, ultra fast speed!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Complaint email no reply 30 divas pachi aavyo ke issue resolved che, wow instant reply!",
    "Urgent grievance mail karyo hatu ane 40 divas pachi automated bot reply aave che, outstanding agility!",
    "Issue resolve thayo nathi pan ticket automatically closed kari didhi, revolutionary problem solving!",
    "Customer service portal ticket close kari ne survey mangyu, 10 out of 10 for imagination!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Technician visit mate 10 AM to 1 PM no time apyo ane aavyo 5 divas pachi, punctual masters!",
    "Repair person bole hu 10 minute ma aavu chu ane aakho divas phone switch off karyo, divine punctuality!",
    "Home repair service booking kari hati, aaje 10 divas thaya technician no पत्तो nathi, five star commitment!",
    "Appliance service team bole 24 hours ma aavishu, aaje mahino puro thayo, supersonic promptness!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Chatbot sathe 1 hour chat karyo ane end ma kidhu 'I did not understand', AI miracle!",
    "Automated support bot constantly loop ma same answer repeat kare che, futuristic intelligence!",
    "AI customer assistant e query solve karva badle FAQs nu link paste kari didhu, mind blowing innovation!",
    "Virtual assistant bole hu tamari madat karva taiyar chu pan ek pan command samajh nathi padti, peak AI!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Food delivery app e coupon code lagavya pachi delivery fee 3 times vadhaari didhi, generous savings!",
    "Free delivery promotion offer ma 100 rupiya packaging charge apply karyo, unbeatable discount!",
    "Discount code apply karyo ane final total original karta pan vadhi gayu, corporate generosity!",
    "50% off coupon code thi 20 rupiya no benefit malyo ane 50 rupiya platform fee, masterstroke savings!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Cake delivery ma photo cake mangavyo hatu ane face completely smudged aavyo, artistic masterpiece!",
    "Birthday cake order karyo hatu ane delivery boy e drop kari smash kari didho, ready to eat abstract art!",
    "Celebration cake par spelling 'Hapy Bithday' lakhi ne moklyo bakery vala e, Nobel prize in literature!",
    "Pastry mangavi hati ane melted soup mali gayo box kholya pachi, premium liquid dessert experience!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Replacement request kari toh bole broken item courier thi amne vapis moklo mara kharche, fair play!",
    "Defective product na return shipping charges customer na matha par nakhya, legendary policy!",
    "Return policy ma lakhyu che 7 days pan request submit karta error aave che, infallible tech infrastructure!",
    "Warranty claim mate bill upload karyu toh system bole file invalid che, flawless automated system!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

# Infrastructure, Road & Urban Life Scenarios (G16 - G25)
add_group([
    "Road par khaddha etla mota che ke swimming pool banavi sakay, smart city planning!",
    "Smart city na road par khaddha evaa che ke roller coaster ride ni jarur j nathi, free amusement park!",
    "Navo road banavya na 2 divas ma 50 khaddha padi gaya, high engineering standards!",
    "City highway par etla gadda che ke gaadi thi vadhare aakash ma udva nu man thay, world class infrastructure!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Varsad ma rasta par 3 foot pani bharai gayu, Venice na gondola ni feel aavi gai Ahmedabad ma.",
    "Bapore varsad aavyo ane street par boat chalavvi pade evi situation thai, Gujarat tourism at peak!",
    "Heavy rain pachi road par river rafting start thai gai, local municipal corporation rocks!",
    "Monsoon season ma rasta samudra bani gaya, free beach front luxury view ghare betha!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Bapore 12 vagye street light chalu che ane raate 9 vagye andharu, power saving at its peak!",
    "Dhur dhoop ma solar street lights full chalu che ane raate andhakar, energy conservation genius!",
    "Midday sunshine ma high-mast lights illuminate thay che, civic responsibility level 100!",
    "Full afternoon heat ma street lights chalu ane night ma dark streets, brilliant governance!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Road banavya pachi tarat pipeline mate aakho road todi nakhyo, coordination level masterclass!",
    "Navo tarmac road banavyo ane 24 hours ma sewage pipe mate ditch khodi nakhyo, civic coordination award!",
    "Asphalt layering thaya na biji divase gas pipeline walo khodan karyo, synchronous town planning!",
    "Fresh road resurfacing karyu ane biji savare cable wiring mate JCB aavi gai, telepathy at work!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Metro construction 10 varsh thi chale che, potana pota na lagan ma metro ma javani maja aavshe.",
    "Metro bridge banavta banavta aakhi generation pass thai gai, heritage architecture project!",
    "Flyover project deadline 5 varsh pehla expire thai gai hati toh pan kaam ongoing che, timeless engineering!",
    "Underpass construction 4 years thi chalte chalte monument bani gayo che, architectural heritage!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "sentiment_contrast")

add_group([
    "Parking ticket 100 rupiya lidha ane gadi dhoop ma unshaded ubhi rakhvi padi, premium hospitality!",
    "Mall ma parking charge cinema ticket karta vadhare che ane gaadi open mud ma park kari, royal luxury!",
    "Commercial complex e 200 rupee valet parking charge karyo ane window open chhodi didhi, ultra secure!",
    "Covered parking board lagavi ne dhoop ma 40 degree heat ma car park karavi, golden service!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Toll plaza par fastag lane ma 45 minute line ma ubha rahya, electronic transparency at speed of light!",
    "Expressway toll par 1 hour queue ma wait karyo, express speed ni feel aavi gai!",
    "Cashless automatic toll plaza par scanner bagdi gayu ane 2 km lambi line lagi, high speed highway!",
    "Superfast highway toll par 40 minute wait karavya pachi receipt na paper khatam thai gaya, ultra efficiency!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Traffic signal 180 second red ane 5 second green rahe che, equality in timing!",
    "Signal par green light 3 second ma yellow thai jaay che, Olympic sprint test for drivers!",
    "Crossroad par 3 minute wait karyo ane green light 4 second ma vanish thai gai, traffic management prodigy!",
    "Red light timer 240 seconds ane green timer 6 seconds, mathematically balanced traffic engineering!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Road par speed breaker etlo uncho banavyo che ke Everest climb kari lidho hoy evi feel aave.",
    "Local society e road par pahad jevo speed breaker mukyo che, suspension durability certification test!",
    "Rasta par speed bump car nu engine touch kari jaay che, aerospace launchpad construction!",
    "Gali ma 10 foot uncho speed bump banavyo che, off-road racing track simulation!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Highway par gayo vachhe bethi che ane police horn marva par fine kare che, wildlife sanctuary on road!",
    "Four lane expressway par dhor aam tem fare che ane horn marva thi fine aave, biodiversity protection!",
    "Main ring road par cattle roaming comfortably ane vehicles 5 km/h par chalte che, eco-friendly highway!",
    "City bridge par 20 cows relax kare che ane vehicles fasaai gaya, peaceful coexistence at peak!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

# Corporate, Career & Office Sarcasm (G26 - G35)
add_group([
    "Yearly appraisal ma 2% hike malyu, have Forbes billionaire list ma name aavshe maru.",
    "Annual increment 1.5% aapi ne boss bole work hard next year, generational wealth creation!",
    "Inflation 8% che ane salary hike 2% aapi, financial independence unlock thai gai!",
    "Appraisal letter ma 500 rupiya hike aapi ne company bole stay motivated, life changing bonus!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "sentiment_contrast")

add_group([
    "Saturday night 9:30 PM e boss no message aavyo urgent meeting che, work-life balance level god!",
    "Sunday morning 8 AM e manager call kare che production release mate, serene work-life equilibrium!",
    "Weekend par family outing ma boss no call aave urgent PPT submit karo, heavenly boundary management!",
    "Friday midnight e email aavyo urgent deliverable Monday 9 AM joye, employee wellbeing champion company!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Company e bonus ni jagya par samosa ane frooti aapi, fortune 500 level rewards!",
    "Diwali bonus ma ek box soan papdi ane plastic cup mali, extraordinary corporate generosity!",
    "Record profits achieve karya pachi company e pen ane dairy reward aapi, unmatched recognition!",
    "Annual milestone celebration ma company e water bottle gift aapi, overflowing corporate gratitude!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Sprint deadline 2 week thi ghataadi ne 3 divas kari nakhi, time travel engineering at its best!",
    "Manager e project delivery date advance kari ane features double kari nakhiya, agile sorcery!",
    "Development timeline reduce kari ane scope 4 times vadhaari nakhyo, realistic project governance!",
    "Two month scope aapi ne bole kal demo apvano che, pure software witchcraft!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Office meeting 2 kalak chali ane conclusion aavyo ke biji meeting rakhishu, productivity milestone!",
    "Two hour status call chali just to decide ke next call kyare rakhvi, peak corporate accomplishment!",
    "Meeting mate pre-meeting ane post-meeting alignment sync rakhi, corporate marathon excellence!",
    "Status report call ma 3 hours waste karya just to read slides aloud, intellectual breakthrough!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Senior dev bole 'code is self-documenting' ane ek pan comment nathi aakha repository ma, poetry!",
    "Senior engineer bole documentation ni jarur nathi code clean che ane 1000 lines spaghetti code che, genius!",
    "Architecture diagram mangyu toh bole mental model che documentation obsolete che, legendary insight!",
    "Legacy codebase ma variable names a, b, c, d che ane lead bole clean architecture principles, masterclass!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "rhetorical_ridicule")

add_group([
    "Reply-all ma 500 loko e 'thanks' lakhyo, inbox clean karvani maja aavi gai aaje.",
    "Company wide email thread par badha 'please remove me' reply-all kare che, communication symphony!",
    "Simple notification email par 200 people e reply-all 'noted' karyo, mail server stress test!",
    "All-employee announcement thread par endless congratulations spam thayo, inbox zero dream come true!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Manager micro-management kari ne screen recording software install karave che, trust environment rocks!",
    "Mouse movement tracker install karyo laptop par, culture of radical empowerment!",
    "Every 10 minute ma manager message kare 'what are you working on', supreme autonomy!",
    "Webcam always on policy implement kari office management e, high trust high performance organization!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "HR bole we are a big family here, pan appraisal time aave tyare strangers bani jaay che.",
    "HR induction ma we are like a family kidhu ane probation pachi pink slip pakdavi didhi, family values!",
    "Company culture statement ma empathy word 20 vakhat lakhyo pan sick leave reject kari, family warmth!",
    "Corporate family claims kare che pan weekend overtime compensation zero aape che, unconditional love!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "rhetorical_ridicule")

add_group([
    "Office coffee machine ma thi lukewarm colored water nikle che, barista cafe experience in house!",
    "Pantry ma instant coffee pouch khatam thai gaya ane warm muddy water aave che, gourmet beverage lounge!",
    "Automated vending machine garam pani ane powder mix nathi karti, premium beverage tasting facility!",
    "Corporate coffee station par thi detergent jevo swad aave che, five star culinary beverage!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

# College, Academic & Student Life Sarcasm (G36 - G45)
add_group([
    "Class ma 75% attendance mandatory che pan teacher pote mahina ma 4 divas aave che, role model!",
    "Professor attendance ma 1 minute late entry allow nathi karta pan pote 30 minute late aave, punctuality hero!",
    "Strict attendance rule che pan syllabus 2 lecture ma fast-forward kari didho, educational excellence!",
    "Teacher absent rahe che full week pan students na attendance shortage par notice aape che, moral leadership!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Group project ma badhu kaam me karyu ane credit aakha group ne malyu, communist equality at its best!",
    "Team assignment ma 3 partners e phone pick na karyo ane presentation day par credit mangva aavi gaya, comradeship!",
    "Final year project report me aakhi night jagine lakhi ane baki badha certificate par sign karva aavya, team spirit!",
    "Project deployment coding solo karyu ane viva ma partner bolyo 'we designed it together', peer bonding!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Assignment copy na karta kidhu hatu ane sir pote Wikipedia copy-paste kari ne slides banave che, original research!",
    "Plagiarism policy par strict lecture aapyo ane exam paper 10 year old paper mathi copy karyo, academic rigor!",
    "Faculty bole internet reference na vaparsho pan handouts seedha Google first result che, scholarship master!",
    "Teacher research integrity par bhashan aape che ane seminar notes uncredited blog mathi lidha, inspiration!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "rhetorical_ridicule")

add_group([
    "Sir e viva ma out of context questions puchhi ne fail karyo, intellectual justice delivered!",
    "External examiner e syllabus bahar no formula puchhi ne zero mark aapi didha, objective assessment!",
    "Viva ma practical project regarding puchhva badle random personal questions puchhiya, rigorous evaluation!",
    "Practical viva ma teacher personal ego satisfy karva question twist karyo, academic enlightenment!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "College library ma shanti rakhva sign board lagavyo che ane librarian pote mobile par reel jova ma busy che, zen mode!",
    "Silence please board niche betha betha peon ane staff jor thi gupshup kare che, library tranquility!",
    "Reading room ma study karva betha ane clerk phone par loud speaker par vaat kare che, peaceful temple of knowledge!",
    "Digital library computer laboratory ma fans no loud ghadghad aawaj aave che, serene studying environment!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "College wifi 10 kbps speed aape che, Silicon Valley level infrastructure in campus!",
    "Campus high speed broadband connect thay che pan Google search par 5 minute buffer thay che, 5G campus!",
    "High tech university wifi connection constantly request time out batavse, digital revolution in education!",
    "Computer lab internet connectivity 2G speed thi pan slow chale che, bleeding edge research facility!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Bhai exam mate 10 minute pehla syllabus puchi rahyo che, top karvani full taiyari che lyo!",
    "Exam hall bahar ubha rahi ne dost bole 'aa subject ma su aave che bhai?', academic dedication!",
    "Semester examination na 15 minute pehla classmate formula sheet magi rahyo che, gold medalist attitude!",
    "Midsem exam start thava na 5 minute pehla friend bole book nu name su hatu, intellectual curiosity!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "rhetorical_ridicule")

add_group([
    "Bhai e party aapis kidhu hatu ane bill mara matha par faadi didhu, true brotherhood in action!",
    "Dost bole treat mara taraf thi ane wallet room par bhuli gayo kidhu, unconditional friendship!",
    "Celebration cafe ma boli ne order double karyo ane payment QR mara phone thi scan karavyo, legendary generosity!",
    "Treat aapis boli ne luxurious dinner karyo ane bill time toilet ma slip thai gayo, friendship goals!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Haa bhai tu j sachho, aakhi duniya toh murkh che tari samne.",
    "Haa bhai tari logic toh Einstein karta pan dangerous che, aakhi duniya galat che tu eklo smart!",
    "Tamari argument itli supreme che ke logic pote surrender kari didhu bhai, hats off!",
    "Tu j universal truth no messiah che, baki badha toh bewakoof betha che sansaar ma!"
], "Sarcasm", "programmatically_authored", "gujarati_romanized", "rhetorical_ridicule")

add_group([
    "Tame toh mota celebrity thai gaya have, amari jeva aam mitra ne kya thi yaad karo.",
    "Bhai have toh multinational star bani gaya lage che, phone upadva no time nathi amari sathe!",
    "Tamaro standard itlo high thai gayo ke message no seen kari ne reply nathi aavto, royal highness!",
    "Aap toh VIP league ma pravesh kari gaya, juni mitrata toh have commoners ni vaat che!"
], "Sarcasm", "programmatically_authored", "gujarati_romanized", "rhetorical_ridicule")

# Social, Tech & Everyday Irony (G46 - G75)
add_group([
    "Gym nu annual membership lidhu ane 2 divas pachi sofa par betha betha chips khavani maja aave che, athletic mindset!",
    "Fitness transformation resolution lidho ane 3rd day par bedtime jalebi khadhi, iron discipline!",
    "Gym protein shaker kharidyo ane andar cold coffee banavi ne ghare sui rahya, ultimate fitness icon!",
    "Yearly workout membership fee bhari ane aaje 6 mahina thi gym nu darwajo nathi joyo, solid financial fitness!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Tari smile etli fake che ke plastic pan sharmai jaay bhai.",
    "Tamaro professional smile photo filter thi pan vadhare synthetic laage che bhai, true authenticity!",
    "Tamari courtesy etli artificial che ke cardboard cutout pan vadhare real lage, natural elegance!",
    "Itlo fake polite tone to automated telephone operator pan nathi vaparti, peak genuineness!"
], "Sarcasm", "programmatically_authored", "gujarati_romanized", "rhetorical_ridicule")

add_group([
    "Bhai e Instagram par workout na 10 photo post karya ane gym ma ek pan dumbbell touch na karyo, fitness influencer!",
    "Mirror selfie 50 khinchi gym ma pan cardio zero minute karyo, pure dedication to aesthetics!",
    "Gym wear pehri ne post mukyo 'sweat today smile tomorrow' ane smoothie piine return thai gayo, true athlete!",
    "Fitness reel record karva 2 hours lagya pan bench press ma empty rod pan uplift na thayo, powerhouse!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Udhar lidhela 5000 rupiya mangya toh bolyo bhai hu travel kari rahyo chu signal nathi aavto, international banker!",
    "Loan vapis magya toh friend bole financial crunch che ane biji post Goa party ni mukhi, legendary integrity!",
    "Paisa parat aapo kidhu toh bolyo UPI pin bhuli gayo chu, world class banking innovation!",
    "Cash udhar lidho hatu ane have calls reject kari ne status par 'investing in peace' lakhe che, financial genius!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Relative e call kari ne puchhyu package ketlu che ane kidhu mara dikra nu toh aa thi 4 times che, affectionate relatives!",
    "Kaka call kari ne marksheet puchhe che ane son na GPA na praise start kare, pure unconditional family love!",
    "Cousin na wedding ma aunties e package ane weight banne par survey start karyo, wholesome community support!",
    "Relatives visit karyo ane advice aapi ke beta tame hard work nathi karta, endless inspiration from elders!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Birthday wish 1 mahino late aavyu ane niche lakhyo 'belated but true wishes', touching emotional bond!",
    "Janmadivas na 45 divas pachi friend no text aavyo 'hbd bro caught up with work', priority friendship!",
    "Anniversary wish ek mahina pachi aavi with excuse ke network lag hatu, high bandwidth camaraderie!",
    "Birthday thaya pachi 3 weeks pachi message 'missed it bro but blessings are permanent', legendary dedication!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Message par 'typing...' 20 minute thi dekhay che ane message aavyo 'k', suspense of the millennium!",
    "WhatsApp par 30 minute typing indicator aavya pachi single thumbs-up emoji send karyo, literary genius!",
    "Half an hour live typing display thayu ane text aavyo 'ok', supreme communicative masterpiece!",
    "Typing bubble continuous 25 minutes chalya pachi dot '.' aavyo, unmatched conversational suspense!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Battery 100% thi 10 minute ma 10% thai gai, outstanding battery backup technology!",
    "Full charge karelo smartphone call chalu karta j 2% warning aapi didhi, flagship battery durability!",
    "Charger disconnected karya na 5 minute ma mobile shutdown thai gayo, unlimited battery life!",
    "New brand phone ni battery airplane mode ma pan drain thai jaay che, futuristic energy optimization!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Software update karyo ane touch screen freeze thai gai, modern technological evolution!",
    "OS update install thaya pachi camera application load thati bandh thai gai, revolutionary stability!",
    "Firmware update notice aavyo ane fingerprint scanner permanently disabled thai gayo, pinnacle of engineering!",
    "Security patch download karyo ane phone random reboot loop ma fasai gayo, seamless user experience!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Alarm 5 vagya no mukyo hatu ane aankh 11 vagye khuli, discipline level unmatched!",
    "Early morning 4:30 AM alarm snoozed 10 times ane bapore 12 vagye bed mathi niklya, relentless routine!",
    "Morning jog mate 5 AM alarm vishwas thi lagavyo ane lunch time par jagaay gayu, legendary willpower!",
    "Study session mate 6 AM alarm setup karyo hatu ane direct afternoon session chalu thayo, morning productivity!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Phone pani ma padi gayo, water resistance test successfully failed, maja aavi gai!",
    "IP68 waterproof phone rain drop ma blank display thai gayo, waterproof technology rocks!",
    "Waterproof rated smartwatch sink ma padta j dead thai gai, miraculous marine endurance!",
    "Waterproof phone screen par pani no tapko padta j short circuit thai gayo, supreme moisture protection!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "sentiment_contrast")

add_group([
    "Smart watch e heart rate 200 batavyo chair par betha betha, medical innovation at peak!",
    "Fitness tracker step count 15000 batavyo bike chalavta chalavta, intense physical fitness tracking!",
    "Resting heartbeat 210 bpm display kare che gadget ane sleeping report accurate che bole, bio-sensing marvel!",
    "Desk par betha betha smartwatch congratulations message ape che for completing 10km marathon, high precision sensors!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Laptop fans etla jor thi chale che ke helicopter take off karse lage che, aerodynamic cooling system!",
    "Browser tab open karta j cooling fan jet engine jetli aawaj kare che, luxury acoustic engineering!",
    "Notebook fan noise etli loud che ke conference call par sound cancelation failed thai gayu, turbo jet engine!",
    "Single PDF read karta laptop ventilator sound vacuum cleaner jevi thava mande che, stealth thermal solution!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "WhatsApp update karyu ane app crash thava mandyu, thank you developers for this feature!",
    "Messaging app updated version download karya pachi contacts delete thai gaya, cutting edge enhancement!",
    "Social media app update karyo ane chat history blank thai gai, breakthrough modernization!",
    "Update install karta application launch j na thaye evi system security banavi che, zero bug policy!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "What a fantastic movie, I slept peacefully throughout the entire second half, cinematic masterpiece!",
    "Picture etlu thriller hatu ke theatre seat par sweet dream enjoy kari shakya, Oscar award screenplay!",
    "Film script etli engaging hati ke audience phone screen scroll karva lagyo, blockbuster entertainment!",
    "Three hour film dialogs etla profound hata ke theater mathi badha half time ma nikli gaya, box office magic!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Flight canceled, luggage lost, and no hotel provided, minor inconvenience by airline!",
    "Airport transfer bus cancel ane baggage misplaced, small joyful hiccups during vacation!",
    "Airline stranded passengers for 14 hours with no water or meal, five star passenger care!",
    "Transit flight missed thai gai ane airline executive bole hotel booking potana pocket thi karo, warm hospitality!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "understatement_irony")

add_group([
    "Traffic police e helmet hatu toh pan pollution mate fine mari didho, target full karvani duty!",
    "Every document valid hatu toh pan number plate dirty che aam kahi chalan kaapi didhu, exemplary law enforcement!",
    "Traffic constable documents verify kari ne smile sathe fine receipt aapi didhi, civic revenue optimization!",
    "Signal green hatu toh pan stopping line touch thai aam kahi fine banavyo, razor sharp legal enforcement!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Garmi 46 degree che ane light chali gai, full AC feel aave che.",
    "AC bandh che ane temperature 45 degree, natural sauna bath mali rahyo che ghare betha.",
    "Bahar 44 degree tapman ane electricity cut, winter season aavi gai lage che.",
    "Room ma 45 degree temperature che ane fan bandh che, luxury wellness resort feel!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "sentiment_contrast")

add_group([
    "Car wash karavya na 5 minute ma varsad aavyo ane kichad lagi gayu, money well spent!",
    "Gaadi clean karavi ne garage thi bahar kadhi ane direct dhoor udi, divine timing!",
    "Bike polish karavya na 10 minute pachi badhe chhikta padi gaya, aaje no divas best che.",
    "Showroom finish car wash karavya pachi varsad ma fasaai gaya, perfect investment!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "sentiment_contrast")

add_group([
    "Rain jacket pehri ne niklyo toh pan andar thi bheeji gayo, waterproof technology rocks!",
    "Chhatri lidhi hati toh pan aakho bhinjai gayo varsad ma, top quality umbrella manufacture!",
    "Waterproof shoes ma pani bharai gayu pehla 5 minute ma, amazing waterproof claims.",
    "Waterproof rain coat pehrya pachi aakho shirt soaked thai gayo, divine fabric technology!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "sentiment_contrast")

add_group([
    "Ghar nu kitchen pipe leak thai rahyo che ane plumber bole cement dry thava 4 divas aapo, water theme park at home.",
    "Bathroom pipe burst thayo ane repair man bole Monday aavish, free swimming pool facility!",
    "Water leakage repair karva lidhu ane aakho hall pani ma dubi gayo, engineering masterpiece.",
    "Tap repair walo 4 divas thi aavu aavu kare che, private waterfall at home!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Dinner ma khavanu bahi gayu, masterchef quality nu food malyu aaje.",
    "Pizza order karyo hatu ane suki papdi jevi crust aavi, culinary brilliance!",
    "Zomato e 2 kalak pachi thandi roti deliver kari, gourmet five star dining at home.",
    "Food completely burnt aavyu ane restaurant bill five star charge karyo, royal chef talent!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Bus conductor e 50 rupiya no change na aapyo ane bole mandir ma daan samji lo, wah re spirituality!",
    "Conductor change ma 5 rupee ni chocolate aapi gayo, modern banking currency substitute!",
    "Ticket conductor change nathi aam kahi ne aagal vadhi gayo, honesty at its peak.",
    "Change na badle toffee aapi didhi, alternative banking currency zindabad!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Society na security guard bhai gate par sui gaya che, amara flat ni safety bhagwan bharose che.",
    "Guard bhai chair par ghuri rahya che ane thief enter thai jaay, high security zone!",
    "Gate par security guard mobile game rami rahyo che, Z-plus security feel aave che.",
    "Apartment security gate unmonitored che ane guard reception ma chai pi rahyo che, ironclad safety!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Society na lift ma 20 minute fasaai gaya, free VIP meditation retreat mali gayu aaje.",
    "Elevator 3rd floor par bandh padi gayu 30 minute, spiritual mindfulness session!",
    "Lift bandh thai gai ane fan pan bandh, free sauna therapy building taraf thi.",
    "Lift breakdown thaya pachi alarm button e pan aawaj bandh kari didhi, calm quiet atmosphere!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "sentiment_contrast")

add_group([
    "Online clothes mangavya medium size na ane elephant fit aavya, tailoring precision outstanding!",
    "T-shirt large size order kari hati ane tent jetli size aavi, perfect fitting standards!",
    "Online jeans 32 size mangavi ane 40 size delivery aavi, accuracy 100 out of 100.",
    "Shirt medium size kidhu hatu ane parachute jevu delivered thayu, tailoring masterclass!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Stock market ma badha paisa dubi gaya, profit j profit che aaje.",
    "Trading ma portfolio 80% down che, financial guru bani gaya chia have.",
    "Share market ma loss karyo ane dost bole portfolio diversify thayu, masterclass logic!",
    "Demat account empty thai gayu ane dost bole long term wealth creation che aa."
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "sentiment_contrast")

# Additional Sarcastic Scenarios (G76 - G125)
# Transit, Rides & Commuting Sarcasm (G76 - G85)
add_group([
    "Ola cab driver call kari ne puchhyu kya javu che ane suni ne cancel kari nakhyo, customer priority ride!",
    "Uber driver pickup point thi 200 meter door ubho rahi ne bole 'aavi jao', doorstep pickup luxury!",
    "Cab driver ride accept kari ne opposite direction ma chalva lagyo, navigation excellence!",
    "Auto driver metre chalu karva badle bolyo double bhadu thase, honest transparent fares!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Train ticket WL 250 hati ane chart preparation pachi cancel thai gai, holiday trip ekdum secured!",
    "Tatkal ticket booking time IRCTC payment gateway crash thai gayo, seamless digital booking experience!",
    "Railway platform par announcement thayu train platform 1 par aavse ane aavi platform 6 par, fitness training for passengers!",
    "Train departure time thi 4 kalak pachi platform par aavi, clockwork precision railway timing!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Baggage carousel par suitcase 2 kalak pachi aavyo ane zip tooti gayeli hati, VIP luggage handling!",
    "Airport luggage belt par bag no handle tutelo malyo, delicate cargo transportation standards!",
    "Airlines suitcase mathi branded perfume gayab kari didhi, complimentary luggage lightening service!",
    "Suitcase completely crushed condition ma delivery mali carousel par, fragile care at its highest!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Bus ticket AC ni lidhi hati ane AC bandh hatu, natural open air ventilation benefit at full price!",
    "Sleeper bus ma window lock tootelu hatu ane dhool aakhi night modha par aavi, desert safari experience!",
    "Luxury sleeper coach bus ma cockroaches ghumta hata, wildlife discovery tour on wheels!",
    "Intercity bus route 3 hours no hatu ane tyre puncture thava thi 8 hours lagya, peaceful scenic journey!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Google Maps e short route batavyo ane dead end ditch ma laavi ne ubho kari didho, artificial intelligence milestone!",
    "GPS navigation e wrong turn batavi ne kichad bharela raste fasaavi didha, masterclass navigation algorithm!",
    "Maps app bole you have arrived at destination ane samne eent ni diwal ubhi hati, accurate spatial mapping!",
    "Navigation audio bole continue straight ane aagal nadi ma pul j na hato, divine algorithmic guidance!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "EV scooter full charge hatu ane highway par 2 km pachi battery zero thai gai, sustainable future transportation!",
    "Electric bike charging time 8 kalak lidho ane range 15 km mali, revolutionary green mobility!",
    "Green mobility vehicle drive karta karta motor overheat thai ne bandh padi gai, eco friendly innovation!",
    "Smart EV display restart loop ma fasaai gayu ane road vachhe lock thai gayu, digital automotive masterpiece!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "sentiment_contrast")

add_group([
    "Valet parking vala bhai e new car no bumper scratch kari ne chavi paachi aapi, free tattoo for the car!",
    "Hotel valet car park karva lidhi ane side mirror todi nakhyo, professional five star care!",
    "Valet driver car nu stereo full blast par mukine chill karto pakdayo, personalized automotive stewardship!",
    "Valet parking attendant e seat settings ane AC vents aam tem fari nakhiya, complimentary cabin customisation!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Metro train automated door closing alarm e hathe thi bag dabavi didho, smart transit safety!",
    "Metro coach AC winter ma freeze kare ane summer ma warm hawa feke che, climate controlled comfort!",
    "Smart card balance deduct thai gayu pan turnstile gate open na thayo, generous automated philanthropy!",
    "Platform display batavse metro in 2 minutes ane 20 minute sudhi train aavti nathi, quantum time relativity!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Railway station waiting lounge ma fan bandh ane mosquitoes no attack, royal passenger hospitality!",
    "VIP passenger waiting hall ma smell evi aave che ke bahar platform par ubhu rehvu saru lage, luxury amenities!",
    "Executive railway lounge entry fee pay kari pan drinking water water-cooler empty hatu, premium refreshment lounge!",
    "Station waiting room ma benches broken che ane washroom door lock nathi, world class travel hub!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Auto driver speedometer 90 km/h batavse 30 ni speed par ane metre running like bullet train, astronomical mathematics!",
    "Night fare double kidhu ane auto petrol puraava mate 15 minute detour karyo, passenger convenience focus!",
    "Auto driver passenger ne rasto puchhi ne commute chalave che, cooperative public driving venture!",
    "Metered auto ma fare base rate karta 3 times extra charge karyo festival na name par, festive pricing cheer!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

# Entertainment, OTT & Social Media Sarcasm (G86 - G95)
add_group([
    "Ott platform subscription pay karyo toh pan har 3 minute ma unskippable ad aave che, ad-free premium enjoyment!",
    "Streaming video quality 4K payment karya pachi 360p ma buffer thay che, cinematic visual fidelity!",
    "Season finale episode suspense par end karyo ane OTT platform e season 2 cancel kari nakhi, storytelling genius!",
    "Movie app monthly subscription renew thayo pan popular movies rent pay karva magse, customer first monetisation!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "YouTube video 5 minute no hatu ane andar 4 unskippable ad aavya, content to advertisement ratio masterclass!",
    "Short cooking tutorial ma 3 sponsor segments ane 2 app promos dekhadya, culinary knowledge summit!",
    "Educational video click karyo ane 15 minute gambling app endorsement chali, ethical digital creator!",
    "Documentary clip play karta pehla 20 second insurance ad skip na thai, educational internet freedom!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Influencer advice aape che 'just wake up at 4 AM and buy real estate', profound financial enlightenment!",
    "Social media guru bole failure is just mindset potana rented sports car ma besi ne, philosophical breakthrough!",
    "Online career mentor 50000 no masterclass sell kare che on how to sell masterclasses, infinite wealth generation loop!",
    "Life coach video ma bole don't be poor just manifest money, Nobel prize in economics guaranteed!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "rhetorical_ridicule")

add_group([
    "Live concert ma singer e aakho time lip sync karyo ane ticket 5000 lidhi, authentic vocal performance!",
    "Music festival VIP ticket mangavi ane singer 3 hours late aavyo 15 minute perform karva, punctuality in rockstar culture!",
    "Musical show ma audio echo etli blast hati ke lyrics samajh j na padi, pristine sound engineering!",
    "Live concert stage light bandh thai gai mid show ane DJ pre-recorded tape play kari didhi, electric atmosphere!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Cafe ma 450 rupiya no avocado toast mangavyo ane single dry bread piece aavyo, royal feast!",
    "Artisan artisanal coffee 380 rupees ma mangavi ane instant coffee mix ni smell aavi, luxury cafe sophistication!",
    "Trendy cafe ma burger mini toy jetlo aavyo ane fries separate charge kari, wholesome hearty meal!",
    "Boutique restaurant ma 200 rupee service charge karyo ane menu order table par potaane laine javu padshe, self-serve royalty!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Expensive rooftop restaurant no city view facing open garbage dump hatu, romantic ambiance at its finest!",
    "Candle light dinner book karyo ane background ma generator no dhuwadho ane aawaj hatu, poetic dining atmosphere!",
    "Scenic terrace cafe booking kari hati ane table laundry dry karvana area pase aapyu, panoramic elegance!",
    "Lake view restaurant booking kidhu ane samne building scaffolding dekhati hati, breathtaking natural vista!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Video game release day par 50 GB update aavyo ane start karta j map mathi char character gayab, polished AAA launch!",
    "Full price game buy kari ane multiplayer server opening day thi down che, smooth online community launch!",
    "Game graphic patch install thaya pachi audio mute thai gai, next-gen immersive gaming technology!",
    "Pre-order package gameplay trailer karta 10 times downgrade graphics aapi, honest marketing campaign!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Cinema popcorn 400 rupiya ma lidhu ane aakho cup cold oil ma soaked niklyo, gourmet cinema snacks!",
    "Multiplex theater ma water bottle MRP karta 4 times charge kari, hydrating national economy single-handedly!",
    "Movie hall sound system right speaker crackle karto hatu throughout the show, high definition Dolby experience!",
    "Cinema recliner seat pushback button broke thayo ane 90 degree straight betha betha film joi, orthopaedic movie luxury!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Social media account hack thayo ane support team bole 'please log in to verify ownership', peak security protocol!",
    "Password recovery mail request karyo ane system purano expired password type karva mangse, cyber security masterclass!",
    "Two factor authentication SMS OTP login session expire thaya na 2 kalak pachi delivery aave che, real-time protection!",
    "Account verification photo submit karyo ane automated bot bole identity unclear che, infallible robotic intelligence!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Motivational speaker lecture 2 hours sunyu ane end ma kidhu success requires hard work, ground breaking scientific discovery!",
    "Self-help webinar host bole drink more water to increase coding speed, revolutionary medical advice!",
    "Productivity podcast speaker bole reduce sleep to 2 hours for wealth creation, biohacking miracle!",
    "Leadership workshop certificate malyo without solving a single problem, leadership greatness achieved!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "rhetorical_ridicule")

# Shopping, Bureaucracy, Tech Glitches & Nuanced Sarcasm (G96 - G125)
add_group([
    "Flat 80% discount banner lagavyo ane base price 100% inflate kari didhi, mathematical philanthropy!",
    "BOGO offer ma billing time cashier bole offer expired yesterday, festive retail transparency!",
    "Clearance sale ma torn shirt packaging ma wrap kari ne sale karyo, discount festival joy!",
    "Mega shopping festival ma shipping charge product price karta double karyo, irresistible e-commerce deal!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Official government form submit karva mate 15 signatures mangya ane badha officers lunch break par hata, efficient public administration!",
    "Passport counter officer bole document Xerox blue ink pen thi sign nathi etle rejected, administrative brilliance!",
    "Driving license renewal test ma biometric machine hand detection fail kare che 10 times, seamless e-governance!",
    "Municipal birth certificate download button click karta 404 error aave che, paperless digital governance!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Printer bole replace yellow ink cartridge to print black and white document, printing hardware logic at its best!",
    "Laser printer paper jam error batavse empty tray ma, supernatural hardware intuition!",
    "Wireless printer direct WiFi scan detect nathi karto pan neighbor na device par print kadhe che, generous networking!",
    "Printer driver installation wizard computer ne 4 times restart karave che, lightweight modern peripheral driver!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Bluetooth earphones audio client meeting ma unbind thai gayo ane mobile speaker par blast thai gayo, flawless connectivity!",
    "Bluetooth headphones volume zero par pan static white noise aape che, built-in ambient relaxation sounds!",
    "Wireless buds microphone human voice swallow kari ne street horns record kare che, intelligent noise capture!",
    "Bluetooth pairing request neighbor na TV sathe instantly connect thai gai, promiscuous wireless protocols!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Microwave oven bowl heat kari didho ane food completely baraf thandu rahyo, innovative ceramic heating technology!",
    "Electric induction stove plate overheat thaine error code de che right when water boils, smart kitchen safety!",
    "Toaster bread slice burn kari ne coal black banavi aape che, artisan culinary toasting expertise!",
    "Refrigerator freezer ma ice cream melt thai gai pan vegetables freeze thai gaya, balanced interior cooling!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Smart TV remote battery cover kholva screwdrivers mangse, user centric industrial ergonomics!",
    "Smart television turn on thava ma 4 minute boot animation batavse, tranquil patience building appliance!",
    "TV software update pachi HDMI input ports disable thai gaya, minimalistic display innovation!",
    "Television voice control background chat sambhali ne random channel switch kare che, hyper-responsive listening!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Salon barber modern fade cut kidhu ane aakho head army bald banavi didho, high fashion transformation!",
    "Beard trim light touchup kidhu ane moustache clean shave kari didhi, avant-garde facial styling!",
    "Hair stylist haircut 1000 rupees lidha ane mirror joya pachi topi pehri ne niklvu padyu, runaway runway model!",
    "Hair salon appointment 4 PM no hato ane 6 PM chair par baithaavya, VIP styling treatment!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Diet salad mangavyo 400 rupees ma ane faqat 3 shredded lettuce leaves niklya, lavish nutritional value!",
    "Health food cafe smoothie bowl mangavya pachi artificial food coloring smell aavi, 100% organic authenticity!",
    "Sugar free dessert mangavyo ane normal sweet karta 3 times extra sweet niklyo, diabetic wellness magic!",
    "Keto diet meal bowl ma 90% boiled potato pieces niklya, high protein dietary innovation!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Milk boil thava stove par chadyo ane aankh 1 second bandh karta j aakhi kitchen slab par spill thai gayu, quantum physics timing!",
    "Chai tapela mathi overflow thai gai the exact moment me gas off karva haath badhavyo, supernatural gravitational timing!",
    "Gas cylinder delivery booking 10 days thi wait kare che ane stove cylinder Sunday night khatam thayo, synchronized perfection!",
    "Pressure cooker whistle blowing time counter drop kari didho ane dal ceiling par chipki gai, kitchen decoration!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Wedding photographer couple pose 20 minute hold karavyo ane photo blurry aavyo, professional visual artistry!",
    "Marriage album 1 varsh pachi delivery mali ane bride na sister nu name wrong printed hatu, priceless family souvenir!",
    "Reception video coverage ma audio mute hatu ane faqat loud background music play thay che, artistic silent cinema homage!",
    "Pre-wedding photoshoot photographer drone pond ma crash kari didho, high budget cinema stunts!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Bank KYC verification mate original document physical visit ma 3 times submit karyo toh pan bole not verified, infallible banking safety!",
    "Bank manager bole server down che lunch pachi aavjo ane 4 PM counter close kari didho, banking at its finest!",
    "Cheque clearance 21 days pachi thayo signature match nathi thayo aam kahi, forensic graphology precision!",
    "Debit card re-issue charge deduct thayo pan card address par dispatch nathi thayo 2 months thi, invisible currency distribution!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Job interview rounds 6 clear karya pachi HR bole salary budget match nathi thato, transparent talent acquisition!",
    "Company recruiter 10 years experience mangse 2 years old technology ma, vision for futuristic expertise!",
    "Interview selection letter aavyo ane join date thi 1 day pehla offer freeze announcement aavi, unwavering corporate commitment!",
    "Unpaid internship assignment ma production ready full-stack website banavva kidhu, generous learning opportunity!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Office chai machine sugar level zero option choose karta pan pure sugar syrup dispense kare che, sweet health hospitality!",
    "Canteen coffee machine button press karta hot empty air bahar aave che, molecular gastronomic tea experience!",
    "Tea bag cup ma dip karta j string tooti gai ane leaves water ma spread thai gaya, artisanal loose leaf experience!",
    "Office cafeteria samosa oil ma swim karto hatu, deep fried luxury lipid intake!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Client release deployment na 1 hour pehla badha core requirements change kari nakhiya, smooth agile delivery process!",
    "Product launch day na morning e marketing team UI color theme badalva magse, seamless cross-functional alignment!",
    "Production release Sunday midnight e deploy karavi ane Monday morning bugs par developers ne blame karyo, collaborative ownership!",
    "Demo meeting ma client e unapproved features show karva demand karyo, collaborative partnership at work!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Work password 14 days ma expire thay che with 25 characters special symbols and Latin script, cyber security nirvana!",
    "Corporate authentication app daily 10 times re-login OTP code prompt kare che, uninterrupted workflow efficiency!",
    "VPN token login karta authentication server timeout error aave che, bulletproof remote productivity!",
    "Enterprise SSO portal identity recognize nathi karto pan spam security alert send kare che, impenetrable network defense!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Emergency team meeting call karyo festival celebration eve par, thoughtful humane managerial empathy!",
    "Diwali holiday sandhye urgent task dispatch karyo with deadline next morning, compassionate executive leadership!",
    "New year countdown 10 minute baki hatu ane production sever alerts blast thava lagya, festive corporate celebrations!",
    "Sunday family dinner vachhe manager slack ping kare 'quick sync possible?', respect for personal work-life boundaries!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "IT helpdesk laptop hardware broken screen mate advice aape che 'restart machine and clear browser cache', technical omnipotence!",
    "System trackpad click nathi thato ane support executive Windows format karva kidhu, nuanced diagnostic mastery!",
    "Keyboard spacebar button stick thai gayu ane IT ticket reply 'update graphics card driver', pinnacle of system repairs!",
    "Laptop battery bulge thai gai ane support technician bole battery saver mode enable karo, proactive safety assurance!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Hostel dal transparent yellow hot water jevi aave che, pure hydration and microscopic protein intake!",
    "College mess roti itli hard che ke plate todva mate hammer ni jarur na pade, artisanal iron rich wheat bakery!",
    "Hostel mess sabji ma faqat potato ane water float thay che, authentic balanced nutrition!",
    "Campus canteen rice ma tiny stones niklya, dental enamel durability verification test!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Society WhatsApp group admin 5 AM good morning flower GIF forward kare che, profound philosophical enlightenment!",
    "Apartment community group fake medical remedies forward kare che, unofficial medical council advice!",
    "Residential complex group par unverified political propaganda shares thay che, communal harmony in action!",
    "Flat owners group par water shortage discuss karva badle horoscope share kare che, practical community problem solving!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Balcony clean kari ne dry kari ane biji minute dhool bhari aandhi aavi gai, meteorological blessing!",
    "Weather report e clear sunshine forecast apyo ane unseasonal torrential downpour aavi gayo, supercomputer meteorological accuracy!",
    "Gardening plants clean karya ane neighbor e upper floor thi dirty water dholi nakhyo, thoughtful terrace solidarity!",
    "Fresh clothes washing dry karva mukya ane bird drop target practice kari gayo, biological outdoor enrichment!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Maggi 2 minute noodles claim kare che ane boil thava ma 20 minute lagya, precision culinary physics!",
    "Instant soup packet prepare karta powder lumps dissolve nathi thata, gourmet smooth texture!",
    "Instant ready to eat meal taste plastic wrapper jevo aave che, futuristic culinary synthetic experience!",
    "Microwave popcorn bag burst thaya vager direct smoke fekva lagyo, delicate aroma in kitchen!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Autocorrect mom ne something utterly bizarre change kari send kari nakhyo, linguistic artificial intelligence marvel!",
    "Keyboard auto-suggest words typing context destroy kari ne embarrassment create kare che, predictive typing masterclass!",
    "Speech to text microphone Gujarati English mix sambhali ne alien language type kare che, universal translator milestone!",
    "Auto-correct spelling correction simple dictionary word replace kari profane slang banavi de che, moral text engine!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Vacuum cleaner dust suck karva badle exhaust port thi aakhi room ma blow kare che, atmospheric dust distributor!",
    "Air purifier filter change karyo ane purifier plastic burn smell throw karva lagyo, pure mountain breeze simulation!",
    "Robotic vacuum floor clean karta dog waste spread kari aakho flat paint kari didho, autonomous robotic artistry!",
    "Smart washing machine cycle complete notification 3 hours late aave che, hyperconnected smart appliance ecosystem!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Online cloth sale ma wool sweater mangavyo ane synthetic nylon mosquito net fabric aavya, cozy winter warmth!",
    "Winter gloves order karya ane waterproof claims par ice rain soak kari didho, sub-zero survival gear!",
    "Rain boots order karya ane sole split thaya thi feet completely muddy thaya, adventure footwear pedigree!",
    "Formal office trousers mangavya ane transparent pajama cloth deliver thayu, executive sartorial perfection!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")

add_group([
    "Ahmedabad 47 degree summer afternoon breeze ne local weather anchor 'pleasant sunny weather' bole che, delightful cooling atmosphere!",
    "Heatwave alert ma tourist department walking heritage tour promote kare che, invigorating summer wellness promenade!",
    "May mahina ni blazing sunshine ma asphalt road melt thai ne shoe sole chipki jaay che, free warm foot massage!",
    "Scorching sun ma outdoor bus stop shade without roof metal bench tapeli banavi nakhi, ergonomic urban lounge!"
], "Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "mock_praise")


# ==============================================================================
# PART 2: NON-SARCASTIC SCENARIO CATALOG (125 groups: G126 to G250)
# Sub-categories: genuine_positive, genuine_negative, factual_neutral
# ==============================================================================

# Benchmark Non-Sarcastic Examples (G126 - G130)
add_group([
    "Finally kaam puru thayu, what a relief!",
    "Finally assignment submission puru thayu, what a huge relief!",
    "Aakhar ma kaam complete thai gayu, haash have shanti mali.",
    "Project deployment successfully puru thayu, what a great relief for the whole team!"
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_positive")

add_group([
    "Wah, kharekhar khub saras kaam karyu che, aabhar!",
    "Wah, kharekhar khubh saras kaam karyu chhe, tamari aakhi team no khub aabhar!",
    "Wah, tamaru performance kharekhar prashansaniya hatu, congratulations!",
    "Wah, tamari presentation ekdum clear ane persuasive hati, client impressed thai gayo."
], "Non-Sarcasm", "programmatically_authored", "gujarati_romanized", "genuine_positive")

add_group([
    "Aaje traffic ma be kalak waste thaya, hu ekdum thaki gayo chu.",
    "Aaje rasta par 2 kalak traffic jam ma fasaai gaya, hu khub thaki gayo chu ane sar dukhe che.",
    "Traffic ma be kalak waste thaya etle mood bilkul kharab thai gayo che aaje.",
    "Highway par heavy jam hatu ane 2 kalak commute delay thayo, physically exhausted chu."
], "Non-Sarcasm", "programmatically_authored", "gujarati_romanized", "genuine_negative")

add_group([
    "Exam ma fail thai gayo, have ghar par su javab aapis?",
    "Exam ma fail thayo etle khub dukh ane afsos feel thay che, khabar nathi have su thase.",
    "Maths na exam ma fail thai gayo chu, have re-exam ni taiyari karvi padse.",
    "Hard work pachi pan exam clear na thai ane fail aavyo, deep sadness feel thay che."
], "Non-Sarcasm", "programmatically_authored", "gujarati_romanized", "genuine_negative")

add_group([
    "Aaje somvar che ane office javu padshe.",
    "Aaje somvar chhe ane morning 9:30 AM e office pohchvu padshe meeting mate.",
    "Somvar na divase daily routine pramane office javu jaruri che.",
    "Aaje Monday che ane weekly team meeting schedule kareli che."
], "Non-Sarcasm", "programmatically_authored", "gujarati_romanized", "factual_neutral")

# Genuine Positive Sentences (G131 - G145)
add_group([
    "Food ekdum delicious hatu ane service pan fast hati, majja aavi gai.",
    "Restaurant ma dinner ekdum tasty hatu ane staff attentive hatu, great evening experience.",
    "Khana ekdum fresh ane flavorful hatu, restaurant atmosphere pan quiet ane clean hatu.",
    "Catering food quality khub saras hati ane badha guests e khavanu appreciate karyu."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_positive")

add_group([
    "Tamari help na lidhe project timely submit thai sakyo, khub khub aabhar.",
    "Tamari guidance ane support thi assignment timely complete thayo, thank you so much.",
    "Tamara valuable inputs thi client proposal easily approve thai gayo, sincere gratitude.",
    "Technical guidance badal aabhar, tamera vina aa deadline meet karvi impossible hati."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_positive")

add_group([
    "Exam ma hard work karyu hatu ane distinction rank malyu, khub khushi thai aaje.",
    "Semester examination ma 9.2 CGPA aavyo, mehnat safal thai ane family khush che.",
    "Entrance test ma high percentile aavya pachi dream college ma admission confirm thai gayu.",
    "Hard work pay off thayu ane top rank sathe semester clear karyo, proud moment for us."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_positive")

add_group([
    "Hotel staff bahu polite ane cooperative hatu, rooms clean ane comfortable hata.",
    "Resort ma stay peaceful ane relaxing hatu, hospitality kharekhar five star level hati.",
    "Hotel check-in smooth thayu ane room amenities perfectly maintained hati, very happy.",
    "Accommodation clean hatu ane room service timely available hati, excellent experience."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_positive")

add_group([
    "Aaj no weather ekdum pleasant che, evening walk par javani majja aavshe.",
    "Aaje hawa shital che ane sky cloudy che, garden ma walking karvani maja aave che.",
    "Pleasant weather na lidhe family sathe outdoor walk plan karyo, feeling energized.",
    "Mild breeze ane pleasant temperature thi evening walk ekdum relaxing bani gai."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_positive")

add_group([
    "Doctor e proper diagnosis karyu ane dava thi tabiyat ma ghanu sudharo aavyo che.",
    "Treatment protocol follow karya pachi fever completely subside thai gayo, relief feel thay che.",
    "Physiotherapy sessions thi joint pain ma major improvement aavyu, walking comfortable thai gayu.",
    "Doctor na empathetic treatment ane medicines thi health rapidly recover thai gai."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_positive")

add_group([
    "New phone no camera ane battery performance banne outstanding che, fully satisfied.",
    "Smartphone battery easily single charge par 2 days chale che ane camera crisp photos le che.",
    "Device performance smooth che ane display quality bright ane sharp che, money well spent.",
    "Mobile purchase thi khub khush chu, processing speed fast che ane battery drain zero che."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_positive")

add_group([
    "Placement interview ma selection thai gayu with good package, celebration time!",
    "Campus placement drive ma software developer role offer malyu, family khub proud che.",
    "Dream company ma interview clear thayo ane confirmation letter mali gayu, great start to career.",
    "Selection letter aavi gayu with excellent stipend and benefits, thank God for this opportunity."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_positive")

add_group([
    "Coding hackathon ma first prize malyu, hard work ane team effort rang lavyo.",
    "College technical hackathon ma amari team winner declared thai, unforgettable achievement.",
    "48-hour continuous coding hackathon ma successful prototype build karyo ane trophy jiti.",
    "Hackathon jury e amara project idea ane implementation ne first position aapi, awesome win!"
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_positive")

add_group([
    "Flipkart thi mangavela sports shoes time par safe condition ma deliver thai gaya.",
    "Online ordered package intact condition ma delivered thayu with proper bubble wrap.",
    "Product quality original ane authentic mali, packaging strong hati ane delivery timely thai.",
    "Courier delivery partner polite hatu ane item undamaged condition ma receive thai."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_positive")

add_group([
    "Amazon customer support e 5 minute ma issue resolve kari refund process start kari didho.",
    "Customer care executive empathetic hatu ane cancellation request quickly process thai gai.",
    "Support helpline quickly answered ane damaged item return pickup immediately schedule karyo.",
    "Helpline agent polite way ma ticket resolve kari refund account ma credit karavyo, great service."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_positive")

add_group([
    "Computer RAM upgrade karva thi multitasking smooth thai gai che, performance improve thayo.",
    "SSD storage install karya pachi laptop boot time fast thai gayo ane apps instant open thay che.",
    "Hardware upgrade complete thaya pachi video editing rendering speed double thai gai.",
    "Memory capacity vadhaari etle software hang thavanu bandh thai gayu ane workflow smooth che."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_positive")

add_group([
    "Ghar ma new air conditioner install karyu ane bedroom quiet ane cool rahe che.",
    "Inverter AC cooling fast kare che ane noise level bilkul silent che, sound sleep aave che.",
    "New cooling unit installation clean thayu ane power consumption energy efficient che.",
    "Air conditioner effective cooling aape che afternoon heat ma, family comfortable che."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_positive")

add_group([
    "Navratri festival ma garba ground nu arrangement khub systematic ane safe hatu.",
    "Cultural garba night ma traditional music ane discipline environment khub saras hatu.",
    "Festival celebration ma security arrangement systematic hatu ane security guards helpful hata.",
    "Navratri garba circle ma dancing space ample hati ane evening energetic ane joyful rahi."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_positive")

add_group([
    "Family trip par peaceful weekend spend karyo, fresh memories bani gai.",
    "Weekend getaway par peaceful atmosphere malyu ane city noise thi door nature enjoy karyo.",
    "Family members sathe quiet weekend picnic enjoy kari, relaxed ane recharged feel thay che.",
    "Short trip par beautiful natural landscape ane quality family time spend karyo."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_positive")

# Genuine Negative Sentences (G146 - G160)
add_group([
    "Laptop motherboard short circuit thai gayu ane badho data lost thai gayo, bohot dukh che.",
    "Storage hard drive crash thai gai ane critical documents restore nathi thaya, heavy loss.",
    "Computer sudden shutdown thayo ane hardware damaged declared thayo, repair cost high che.",
    "Internal drive fail thaya thi 5 years na photographs permanently delete thai gaya, sad situation."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_negative")

add_group([
    "Food deliver thava ma 2 kalak lagya ane khavanu completely thandu hatu, zero rating aapis.",
    "Meal preparation bad hatu ane delivery cold food deliver kari, stomach upset thai gayu.",
    "Order deliver thava ma unnecessary delay thayo ane food stale hatu, highly disappointed.",
    "Food packaging leak thai gayu hatu ane curry spill thai gai, unhygienic condition hati."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_negative")

add_group([
    "Mobile screen crack thai gai padva thi, repair center ma kharcho bahu vadhare bole che.",
    "Smartphone unpadded floor par padta touch digitizer shatter thai gayu, expensive replacement cost.",
    "Display glass tootva thi touchscreen work nathi karti ane replacement parts expensive che.",
    "Screen break thaya pachi repair estimate mobile budget karta half aavyo, financially stressed chu."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_negative")

add_group([
    "Train cancel thai gai che cyclone alert na lidhe, amaro tour cancel karvo padyo.",
    "Severe weather alert na karane flight cancelled thai gai ane hotel non-refundable hatu.",
    "Cyclone warning na lidhe trains suspend thai gai ane long planned holiday ruin thai gayo.",
    "Inclement weather conditions na lidhe transport halted che ane vacation booking lost thai."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_negative")

add_group([
    "Internet connectivity drop thaya kare che call ma, urgent interview disconnect thai gayo.",
    "Broadband fiber cable cut thaya thi whole day online working halt thai gayu, frustrating day.",
    "Wi-Fi signal frequently disconnect thay che ane important video meeting miss thai gai.",
    "Unstable internet provider na lidhe online test auto-submit thai gayu, heavy grade loss."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_negative")

add_group([
    "Heavy rain na karane basement ma pani bharai gayu ane car engine damage thai gayu.",
    "Monsoon storm water basement parking ma overflow thayu ane vehicles submerge thai gaya.",
    "Waterlogging basement ma accumulate thayu etle car electrical system destroyed thai gai.",
    "Basement flood thi car submerged thai ane insurance claim assessment ma delays che."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_negative")

add_group([
    "Product defective deliver thayo hatu ane customer care phone pick nathi karto, fraud company laage che.",
    "Broken electronics unit aavyu ane support helpline unresponsive che, cheated feel thay che.",
    "Defective machinery received ane replacement request unanswered che, consumer court complain karish.",
    "Item damaged state ma aavyu ane returns link disabled che, unethical seller behavior."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_negative")

add_group([
    "Job interview ma preparation pachi pan rejection malyu, sad feel thay che.",
    "Multiple assessment rounds pachi final interview ma reject thayo, confidence low thai gayo che.",
    "Months of coding practice pachi pan rejection email aavyo, deep disappointment feel thay che.",
    "Interview clear na thayo ane rejection letter aavyo, future career path regarding worry che."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_negative")

add_group([
    "Bank server down che etle urgent fund transfer na thayo ane fine lagyo.",
    "Banking portal transaction failure na lidhe electricity bill due date miss thai ane penalty lagi.",
    "Server network error thi hospital fee transfer delay thayo, critical stress condition bani gai.",
    "Bank downtime na lidhe time par payment fail thayo ane vendor order cancel kari didho."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_negative")

add_group([
    "Auto driver e meter thi javani na paadi ane double fare mangyo, argument thai gai.",
    "Rickshaw driver standard meter refusal kari illegal surge charge mangyo, heated dispute thayo.",
    "Auto driver night time extra charge insist karyo ane argument escalate thai, unsafe feel thayu.",
    "Metered fare thi transport deny kari arbitrary lump sum amount demand kari auto driver e."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_negative")

add_group([
    "Heavy rainfall na lidhe ghar na roof thi water tapke che, repair contractor sodhvo padse.",
    "Terrace water seepage bedroom ceiling damage kari rahyo che, urgent plastering required che.",
    "Continuous rain thi apartment wall dampness vadhi gai che ane paint peel off thay che.",
    "Roof leakage na lidhe furniture protect karva covers mukva padya, stressful domestic chore."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_negative")

add_group([
    "Salary deduction notice thi monthly budget bagdi gayu, financial anxiety feel thay che.",
    "Unannounced pay cut notice thi essential household expenses manage karva tough thai gaya.",
    "Company payroll salary deduction karyo without clarification, stress level bohot high che.",
    "Deduction circular aavyo ane EMI payment balance disrupt thai gayu, worried about budget."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_negative")

add_group([
    "Laptop battery charge nathi thati ane charger adapter overheat thay che, repair jaruri che.",
    "Charging port loose connection na lidhe laptop intermittent supply le che ane battery drain thay che.",
    "Battery health 20% drops thai gai ane charger cable melt thava mandyu, hazardous device issue.",
    "Power brick excessive heat generate kare che ane laptop shut off thay che, hardware fault."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_negative")

add_group([
    "Food quality unhygienic hati ane stomach infection thai gayu, restaurant viruddh complaint raise kari.",
    "Contaminated food consumption thi food poisoning thayo ane hospital visit karvi padi.",
    "Stale street food khava thi abdominal pain ane dehydration thayo, severe health concern.",
    "Unhygienic kitchen hygiene thi foodborne illness thai, health inspection viruddh report file karyo."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_negative")

add_group([
    "Train 4 kalak late che ane station waiting area ma seats available nathi, exhausted feel thay che.",
    "Railway delay 4 hours extend thayo ane platform overcrowded che, tired standing for long.",
    "Train arrival postponed thai gai ane cold night ma platform par wait karvu difficult che.",
    "Delay announcement continuous vadhti jaay che ane seating arrangement absent che, physically drained."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_negative")

# Factual Neutral Sentences (G161 - G175)
add_group([
    "Flight nu schedule change thayu che, sanje 6 vagye departure che.",
    "Airlines schedule notification pramane revised departure timing evening 6:00 PM che.",
    "Flight departure slot change thaine 18:00 hours confirmed che according to boarding pass.",
    "Air travel timing change notice mali che, departure 6 PM schedule karyo che."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "factual_neutral")

add_group([
    "Me project report submit kari didhi che, kal review meeting schedule kareli che.",
    "Technical documentation portal par upload thai gayu che ane review session tomorrow schedule che.",
    "Project report submission completed che ane evaluation panel kal meeting conduct karse.",
    "Deliverable report submitted officially che ane review meeting team sathe tomorrow hoy che."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "factual_neutral")

add_group([
    "Kal thi Gujarat ma monsoon season start thay che according to IMD forecast.",
    "Meteorological department bulletin pramane state ma monsoon arrival date tomorrow che.",
    "Weather department forecast indicate kare che ke Gujarat ma seasonal rainfall kal thi begin thase.",
    "IMD official notification pramane monsoon rainy spell tomorrow thi start declared che."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "factual_neutral")

add_group([
    "Hu grocery store jaav chu, tamare koi vegetables mangavva che?",
    "Supermarket groceries purchase karva jaine list check karu chu, do you need anything?",
    "Local market thi daily groceries ane kitchen items lavva jaav chu, vegetables jarur che?",
    "Grocery counter par shopping mate niklu chu, send vegetable items list if required."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "factual_neutral")

add_group([
    "Bank of India IFSC code cheque book ane branch passbook par printed hoy che.",
    "Online NEFT RTGS transfer mate branch IFSC code bank document par easily verify thay che.",
    "Bank branch identification code passbook first page par clear format ma mentioned hoy che.",
    "Account statement ane cancelled cheque par standard 11 character IFSC code displayed hoy che."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "factual_neutral")

add_group([
    "Aadhaar card address update online UIDAI portal thi verify kari ne update kari sakay che.",
    "Resident address update portal through valid address proof document upload kari process thay che.",
    "UIDAI official website par demographic details self-service update portal link available hoy che.",
    "Online Aadhaar identity portal OTP verification through residential address record revise kare che."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "factual_neutral")

add_group([
    "Ahmedabad Municipal Corporation e property tax payment mate online portal open karyo che.",
    "Civic property tax digital payment gateway AMC official web application par accessible che.",
    "Municipal corporation municipal tax collection website link net banking sathe functional che.",
    "City civic center tax payment transaction facility citizen portal par active che."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "factual_neutral")

add_group([
    "Python 3 virtual environment creation mate python -m venv command use karvama aave che.",
    "Virtualenv environment setup mate command prompt ma standard python command execute thay che.",
    "Isolated package dependency environment python standard module venv through initialize thay che.",
    "Project workspace folder ma clean python environment build karva terminal command run kariye chia."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "factual_neutral")

add_group([
    "Ahmedabad metro Thaltej to Vastral route par daily services morning 6:30 AM thi operational che.",
    "East West metro corridor passenger trains operational schedule morning 6:30 AM start thay che.",
    "Thaltej railway metro station terminal train dispatch regular interval par morning to night chale che.",
    "Metro rapid transit route schedule daily operational timetable public notice ma specified che."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "factual_neutral")

add_group([
    "Ahmedabad metro APMC to Motera stadium par daily services morning 6:30 AM thi operational che.",
    "North South corridor line APMC thi Motera connectivity transit frequency 12 minutes maintain kare che.",
    "Stadium route metro connectivity timings commuters mate morning to evening scheduled che.",
    "APMC metro terminus line operational departures daily timetable pramane run thay che."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "factual_neutral")

add_group([
    "State Bank of India online banking portal par RTGS timings 24 hours available hoy che.",
    "Electronic gross settlement transfer SBI net banking customer portal par around the clock operational che.",
    "Real Time Gross Settlement banking channel year-round 24x7 operational standard follow kare che.",
    "Online bank account portal round the clock RTGS digital transaction facilitate kare che."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "factual_neutral")

add_group([
    "Railway reservation counters par senior citizen quota verification counter par available hoy che.",
    "Concessional senior citizen passenger ticket computerized reservation center thi issue thay che.",
    "Railway ticket booking counter age proof verification requirements check kari quota allocate kare che.",
    "Senior citizen reservation facility designated counters par official working hours ma process thay che."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "factual_neutral")

add_group([
    "Ahmedabad BRTS corridor par smart travel cards through ticket booking available che.",
    "Janmarg bus rapid transport transit stations turnstiles contactless smart cards scan kare che.",
    "BRTS dedicated corridor par feeder buses route display electronic boards lagavel che.",
    "City public transport travel pass students mate subsidized rate par issue karvama aave che."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "factual_neutral")

add_group([
    "Gujarat University examination circular official domain portal par PDF format ma uploaded che.",
    "External degree evaluation timetable university controller of examination notice ma declared che.",
    "Semester grade cards distribution college administrative counter thi conduct thase.",
    "University research scholar entrance exam syllabus official site par download link sathe che."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "factual_neutral")

add_group([
    "Python pandas library read_csv method through tabular CSV datasets load karvama aave che.",
    "DataFrame groupby method aggregation analysis calculate karva widely utilize thay che.",
    "Scikit-learn train_test_split module machine learning dataset evaluation splits prepare kare che.",
    "Matplotlib plotting functions through confusion matrix heatmap visualize kari sakay che."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "factual_neutral")

# Additional Genuine Positive Scenarios (G176 - G195)
# Real praise, literal appreciation, success, gratitude
add_group([
    "Wah, su mast ras-puri banavya che aaje mom, ekdum maja aavi gai!",
    "Wah, su mast kheer banavi che aaje, ekdum authentic taste che!",
    "Wah, su mast dhokla banavya che mummy, breakfast ekdum perfect bani gayo!",
    "Wah re wah, su mast handvo banavyo che gher, culinary perfection!"
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_positive")

add_group([
    "Client e software release demo ma product quality khub appreciate kari, team mate big milestone che.",
    "Client presentation smoothly conclude thai ane deliverables praise thaya, great achievement.",
    "Customer feedback survey ma 5 star rating mali ane product design applaud karyu client e.",
    "Stakeholder review call ma project architecture appreciate thayo, team motivated feel kare che."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_positive")

add_group([
    "Doctor ni dava ane strict diet follow karva thi dadaji ni health ma rapid improvement aavyo che.",
    "Grandfather's recovery after medical intervention khub fast rahi che, family ma relief spread thayo.",
    "Cardiologist na medication treatment thi blood pressure completely normal range ma stabilized che.",
    "Proper medical care ane compassionate nursing thi patient tabiyat swiftly recover thai."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_positive")

add_group([
    "New laptop no battery backup 10 hours continuous work mate reliably chale che, very dependable device.",
    "Office coding ane design tasks mate laptop hardware performance ekdum powerful ane consistent che.",
    "Lightweight ultrabook daily travel mate convenient che ane processing speed snappy rahe che.",
    "Laptop thermal cooling quiet che ane demanding workload smoothly handle kare che, solid purchase."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_positive")

add_group([
    "Ahmedabad Sabarmati Riverfront par evening walk ekdum relaxing ane calm atmosphere provide kare che.",
    "Riverfront garden walkway serene environment ma physical walking exercise mate best location che.",
    "Cool breeze along river banks evening outdoor stroll peaceful ane refreshing banavi de che.",
    "Riverfront promenade par family sathe peaceful walk karyo, pleasant evening experience."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_positive")

add_group([
    "Customer support representative e polite tone ma difficult issue 10 minute ma effectively resolve kari didho.",
    "Helpline agent courteous hatu ane dispute refund step by step immediately process karavyo.",
    "Customer care executive patient listening thi billing clarification provide kari ane waiver aapyo.",
    "Professional support team inquiry address kari ne prompt written confirmation mail send karyo."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_positive")

add_group([
    "Gir wildlife sanctuary safari trip ma Asiatic lions spot thaya, thrilling ane unforgettable memory bani gai.",
    "National park safari tour exceptionally successful rahyo ane wild animal sightings close thi thaya.",
    "Nature forest tour ma lion pride resting near safari track joya, majestic experience hatu.",
    "Forest conservation safari guide knowledgeable hatu ane flora fauna insights khub interesting hata."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_positive")

add_group([
    "Morning yoga ane meditation session regular rakhva thi concentration level ane stamina ma noticeable growth aavyu.",
    "Daily pranayama ane stretching routines back posture ane mental clarity boost kare che.",
    "Regular fitness workouts energy level active rakhe che ane stress substantially reduce thay che.",
    "Mindfulness meditation practice morning quiet hours ma peace ane emotional stability instill kare che."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_positive")

add_group([
    "Highway par car tyre puncture thaya par local mechanic e prompt assistance provide kari wheel change karyo.",
    "Roadside flat tyre emergency ma passerby driver e help kari spare tyre mount kari didho, grateful.",
    "Expressway break-down assistance service vehicle 20 minute ma arrive thai ne issue fix karyo.",
    "Highway breakdown helpline efficient support dispatch kari safe journey resume karavi."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_positive")

add_group([
    "College library quiet study zone semester exam revision mate perfect silence provide kare che.",
    "University reading section tranquil ambiance deep focused academic study allow kare che.",
    "Air conditioned library reference section collection research papers access karva best space che.",
    "Silent reading floor distraction free environment ma exam syllabus revise karva ideal facility che."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_positive")

add_group([
    "Sister e state civil services examination first attempt ma high rank thi clear kari, proud family moment.",
    "Competitive administrative examination merit list ma sister nu top rank nomination aavyu, ecstatic.",
    "Hard work ane continuous discipline thi competitive entrance exam crack karyo, immense pride.",
    "Civil services qualification letter aavyo ane family celebrations excitement sathe start thaya."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_positive")

add_group([
    "Film direction cinematography ane realistic acting performance kharekhar inspiring hatu, worth watching movie.",
    "Cinema presentation meaningful social message impactful storytelling through convey kare che.",
    "Documentary film background score ane visual editing masterclass standard exhibit kare che.",
    "Theatrical drama emotional depth ane character portrayal through audience ne deeply move kare che."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_positive")

add_group([
    "Passport renewal application verification Police inquiry pachi faqat 4 business days ma dispatch thai gayo.",
    "Passport Seva Kendra biometrics processing efficient hati ane booklet timely receive thai.",
    "Speed post tracking through new passport book doorstep par prompt delivery mali, commendable efficiency.",
    "Regional passport office document review seamless conduct karyo ane expedited issuance delivery thai."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_positive")

add_group([
    "Blood donation camp successfully organize thayo ane 250 units blood safely collect thayu, humanitarian pride.",
    "Community youth volunteers blood donation drive conduct kari emergency pediatric units assist karya.",
    "Red cross society blood bank association ma charitable blood collection milestone achieve karyo.",
    "Voluntary donors camp organized seamlessly ane healthcare professionals gratitude convey karyo."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_positive")

add_group([
    "Solar rooftop panels installation karyo pachi monthly domestic electricity bill 80% reduce thai gayo.",
    "Rooftop solar energy system clean power generate kare che ane grid billing credits benefit aape che.",
    "Solar photo-voltaic cells setup continuous green electricity supply produce kare che, sustainable savings.",
    "Domestic solar power units implementation financial budget save kare che ane carbon footprint mitigate thay che."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_positive")

add_group([
    "Dentist clinic ma tooth extraction procedure completely painless method thi conclude thayo, comfortable relief.",
    "Dental surgeon skilled technique through decay cleaning performed without soreness, grateful experience.",
    "Oral clinic sterilization standards exemplary che ane cavity filling treatment smoothly completed.",
    "Gentle dental practitioner medication advice through rapid gum recovery ensure kari, wonderful care."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_positive")

add_group([
    "Stormy turbulent weather conditions ma pilot e commercial flight smoothly runway par touch-down karavyo, skilful handling.",
    "Heavy crosswinds runway approach par flight captain safe landing precision through execute kari.",
    "Inclement air conditions brave kari aircraft safe destination arrival timely touch down thayo, thankful to crew.",
    "Cockpit flight crew professional composure through turbulent storm navigation secure land karavyo."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_positive")

add_group([
    "Online data science certification successfully complete karyo ane verified credential resume ma add karyo.",
    "Machine learning specialization coursework completed with honors distinction certificate.",
    "Cloud computing certification examination 90% score thi pass karyo, professional competence milestone.",
    "Technical skill assessment program clear kari digital badge profile par integrate karyo."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_positive")

add_group([
    "Society annual sports day cricket match exciting final over finish thi conclude thayo, energetic community day.",
    "Residential complex badminton tournament friendly sportsman spirit sathe celebrate karyo.",
    "Colony cultural sports festival children ane elders sathe enjoyable games activities organize karya.",
    "Neighborhood sports event enthusiastic participation attract karyo ane healthy fitness promote thayo."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_positive")

add_group([
    "Kitchen refrigerator cooling technician prompt inspection kari compressor relay fix kari didho, fast service.",
    "Appliance engineer replacement part authentic invoice sathe install kari cooling restore kari.",
    "Refrigerator technician 1 hour under arrive thai cooling gas leakage resolve karyo, appreciative service.",
    "Authorized service technician polite manner ma domestic freezer issue diagnose kari repair karyo."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_positive")

# Additional Genuine Negative Scenarios (G196 - G215)
# Complaints, real frustration, sadness, authentic negative sentiment
add_group([
    "Alarm snooze button accidental press na lidhe examination hall 30 minute late pohchyo, paper miss thayo.",
    "Oversleeping na karane important job assessment slot lose thai gayo, immense self-remorse feel thay che.",
    "Morning wakeup failure leading to gate closure at university exam center, stressful painful mistake.",
    "Woke up an hour behind schedule ane critical morning interview panel missed, devastated feelings."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_negative")

add_group([
    "Public commute ma wallet lost thai gayu containing original Aadhaar card and driving license, heavy distress.",
    "Bus travel timing pocket theft thayo ane important documents credit cards vanish thaya, lodging police FIR.",
    "Missing purse containing government identity cards ane cash causing overwhelming inconvenience ane worry.",
    "Lost wallet incident na lidhe badha cards block karva padya ane identity re-issuance tedious process che."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_negative")

add_group([
    "Severe migraine headache attack thi office work impossible thai gayu ane dark room ma rest jaruri che.",
    "Intense throbbing head pain sensory sensitivity create kare che, debilitating physical discomfort.",
    "Migraine symptoms onset thi work screen focus fail thayo, compelling medical sick leave.",
    "Severe headache episode accompanied by nausea completely disrupts working schedule, severe pain."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_negative")

add_group([
    "Expressway isolated stretch par car tyre burst thayo ane nearby roadside puncture shop available nathi, stranded.",
    "Night driving highway segment par double flat tyre situation helpless condition create kare che.",
    "Deserted road path par breakdown thayo ane vehicle towing service arriving delay kare che, anxious situation.",
    "Tyre blowout on highway shoulder causes severe anxiety in absence of immediate garage repair assistance."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_negative")

add_group([
    "Critical coding technical assessment submit karti vakhte electricity blackout thayo ane progress lost thai gai.",
    "Power outage during competitive online programming contest disqualified exam session, bitter frustration.",
    "Unscheduled load shedding during terminal submission disrupts computer connection, unfair grade loss.",
    "Sudden black out shut down workstation desktop mid-evaluation, causing irreparable academic distress."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_negative")

add_group([
    "Office rolling chair wheel niche earphone wire squash thayo ane one side audio permanently mute thai gai.",
    "Premium headphones cable sliced by chair casters accidentally, expensive loss due to carelessness.",
    "Audio accessory cord snapped under desk wheel movement, distressing sudden hardware damage.",
    "Headphone wire snapped cleanly beneath chair rollers, ruining audio calls during work."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_negative")

add_group([
    "Seasonal pollen exposure thi severe respiratory allergic reaction trigger thayo, breathing difficulty feel thay che.",
    "Dust allergen reaction causing continuous sneezing, eye inflammation and breathing distress, visiting physician.",
    "Allergy flare up leaves nasal passages blocked and throat irritated, needing antihistamine medications.",
    "Acute allergy symptoms restrict outdoor activities severely, constant fatigue and congestion."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_negative")

add_group([
    "Connecting flight transit point par airline baggage misplaced declared thai gayu, vacation wardrobe ruined.",
    "Luggage failed to arrive at international carousel destination, causing severe disruption and helplessness.",
    "Airport lost baggage counter filing property irregularity report with zero immediate clothes available, upset.",
    "Checked suitcases missing during overseas transit, requiring emergency basic supplies procurement."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_negative")

add_group([
    "Prepaid resort booking hotel management e technical overbooking reason aapi unilaterally cancel kari nakhi.",
    "Advance holiday stay cancelled without prior intimation by lodge proprietors, creating family inconvenience.",
    "Hotel arbitrarily voided accommodation reservation on arrival night, stranding travelers without rooms.",
    "Confirmed vacation booking cancelled unexpectedly, ruining holiday arrangements with non-refundable tickets."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_negative")

add_group([
    "Street storm sewer overflow thi road waterlogged thai gayu, residential building premises exit blocked che.",
    "Severe urban inundation prevents pedestrians from leaving apartment society premises, trapped by dirty water.",
    "Waterlogging surrounding domestic complex entrance submerged footpaths completely, commute disabled.",
    "Street flooding stalls traffic vehicles and inundates entryway gates, causing severe civic helplessness."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_negative")

add_group([
    "Continuous screen glare thi severe eye strain ane burning sensation develop thayo, vision blurry lagti che.",
    "Unrelenting computer display usage triggers ocular fatigue and painful optical redness, need ophthalmologist.",
    "Excessive screen exposure causes acute eye dryness and chronic temple strain, doctor advises screen detox.",
    "Intense monitor hours induce painful eye tension and optical fatigue, requiring protective lenses."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_negative")

add_group([
    "Reading spectacles accidental drop thi frame crack thayo, optician store two days lens fabrication time magse.",
    "Eyeglasses cracked while working and replacement prescription glass requires two day wait, visually impaired.",
    "Prescription frames broke on impact, incapacitating reading ability until new optical set delivered.",
    "Damaged eyesight glasses hamper reading assignments severely, waiting anxiously for optical clinic repair."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_negative")

add_group([
    "High fever ane acute viral infection symptoms exacerbation na lidhe hospital inpatient admission jaruri banu.",
    "Sudden medical condition deterioration needing intravenous drips and hospital monitoring, family under tension.",
    "Severe dengue symptoms require hospitalization for platelet count stabilization, physically weak and worried.",
    "Viral illness requiring hospital bed confinement causes emotional anxiety and physical debility."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_negative")

add_group([
    "Bridge vehicular collision na lidhe massive traffic gridlock thayo, scheduled specialist doctor consultation miss thayo.",
    "Gridlock bottleneck on overpass prevents ambulance passage and delays patient medical checkup appointment.",
    "Dense bridge traffic congestion traps commuter vehicles for hours, forfeiting urgent specialist appointment.",
    "Traffic standstill on major flyover ruins schedule, leading to missed medical consultation."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_negative")

add_group([
    "Bank ATM cash dispense failure kari debit card swallow kari didhu, account debit notification alert aavi gai.",
    "Automated cash machine swallowed card during transaction without dispensing funds, leaving customer stranded.",
    "ATM hardware error detained payment card and money got debited, urgent bank dispute escalation needed.",
    "Cash dispenser detained debit card while withholding requested currency, causing immediate financial panic."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_negative")

add_group([
    "Courier transit ma valuable consignment box crushed condition ma aavyo, inside contents completely damaged.",
    "Shipping package breached during cargo handling and broken fragments received, raising courier claim dispute.",
    "Parcel arrives crushed and fragile contents shattered, courier company refuses immediate accountability.",
    "Mailed fragile glassware arrived in pulverized state due to rough logistics handling, distressing outcome."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_negative")

add_group([
    "Local dairy store mathi purchase karelu milk packet sour ane spoiled niklyo, morning tea preparation spoiled.",
    "Curdled milk packet discovered upon boiling, requiring disposal and ruined breakfast routine.",
    "Packaged dairy pouch contaminated with off-flavor and sour coagulation, causing morning domestic frustration.",
    "Spoiled milk carton purchased fresh from neighborhood store curdles on heating, irritating disruption."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_negative")

add_group([
    "Winter night travel timing jacket main zipper teeth detached thaya, cold winds causing severe shivering.",
    "Coat zipper malfunctioned during chilly night journey, leaving traveler exposed to biting cold winds.",
    "Broken winter garment fastening mechanism lets freezing breeze penetrate, resulting in uncomfortable transit.",
    "Jacket runner slipped off track on overnight bus voyage, shivering through chilly breeze."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_negative")

add_group([
    "Critical unsaved project work laptop sudden thermal blue screen crash thi erased thai gayu, devastating loss.",
    "Operating system crash unceremoniously wiped hours of draft design work, feeling immensely frustrated.",
    "Workstation crash destroyed unsaved documentation right before review, causing intense distress.",
    "Sudden device crash eradicated crucial project calculations, necessitating painful complete rework."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_negative")

add_group([
    "Weekend society friendly cricket match ma fielder slide karti vakhte ankle ligament sprain thayo, painful injury.",
    "Sports field accident caused severe ankle swelling and immobility, requiring plaster cast and crutches.",
    "Twisted ankle during casual sports recreation leaves limb swollen and painful, bed rest mandated.",
    "Ligament tear while playing cricket causes excruciating pain and temporary physical disability."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_negative")

# Additional Factual Neutral Scenarios (G216 - G235)
# Informational, factual statements, logistics, procedural rules
add_group([
    "Gujarat Technological University library reference division official timing evening 8 PM sudhi open rahe che.",
    "GTU central campus library academic book lending counter closes daily at 8:00 PM.",
    "University engineering library catalogue collection evening eight o clock until accessible hoy che.",
    "Academic library reading facilities students mate night eight PM sudhi operational schedule maintain kare che."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "factual_neutral")

add_group([
    "India Post registered speed post consignment delivery standard transit duration 3 business days estimate kare che.",
    "Postal department tracking portal speed post package delivery timeline across major cities display kare che.",
    "Domestic speed post parcels standard transit commitment three working days within dispatch ensure thay che.",
    "Registered postal items destination address par delivery confirmation three days standard follow kare che."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "factual_neutral")

add_group([
    "National Highway 48 stretch par nearest fuel petrol pump 2 kilometers distance par located che.",
    "Highway expressway corridor along petrol dispensing station next exit par situated hoy che.",
    "Fuel filling station facility highway route par two kilometers ahead signage through indicated che.",
    "Motorists refueling service station distance marker highway lane par clearly visible che."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "factual_neutral")

add_group([
    "Election Commission voter ID card address rectification mate valid residential proof document mandate kare che.",
    "Electoral roll voter identity registration online NVSP portal residential documentation verify kare che.",
    "Voter card update procedure identity verification and residential utility bill attachment require kare che.",
    "Election photo identity card revision application standard address proofs support through process thay che."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "factual_neutral")

add_group([
    "Machine learning scikit-learn library classification report precision recall F1-score compute kare che.",
    "Model performance evaluation metrics sklearn metrics module functions through easily evaluate kari sakay che.",
    "Scikit learn library confusion_matrix function class prediction distribution visual breakdown generate kare che.",
    "Supervised text classification benchmarks cross-validation accuracy metrics module compute kare che."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "factual_neutral")

add_group([
    "Domestic RO water purifier sediment filter replacement manufacturer recommendation pramane every 6 months required hoy che.",
    "Water filtration membrane service interval six monthly maintenance cycle adhere karva specify che.",
    "Purifier carbon and sediment cartridges periodic replacement pure drinking water TDS balance kare che.",
    "Household water treatment plant filter cartridges six months periodic renewal recommendation follow kare che."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "factual_neutral")

add_group([
    "College campus cafeteria food menu items list every Monday morning notice board par display thay che.",
    "Student canteen weekly dietary menu schedule beginning of the week circular through publish kare che.",
    "Institutional dining hall weekly food offerings schedule Monday morning regularly bulletin board par update hoy che.",
    "University cafeteria breakfast and lunch preparations weekly menu announcement start of week announce thay che."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "factual_neutral")

add_group([
    "Indian Railways AC classes tatkal reservation booking slot daily morning 10 AM IRCTC portal par open thay che.",
    "Tatkal ticket reservation window for air-conditioned coaches precisely opens at ten AM daily.",
    "IRCTC website railway reservation booking portal 10:00 AM AC tatkal quota allotments enable kare che.",
    "Railway booking guidelines stipulate tatkal AC booking access starts at 10 AM on departure eve."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "factual_neutral")

add_group([
    "Ahmedabad municipal public transit bus route number 13 Kalupur railway terminus to Thaltej connect kare che.",
    "AMTS public transport bus transit schedule route thirteen between Kalupur and Thaltej crossway ply kare che.",
    "City bus network route 13 commuter transit connecting railway junction to western suburbs serve kare che.",
    "Local urban bus line 13 passenger transport service daily timings schedule regular transit maintain kare che."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "factual_neutral")

add_group([
    "Electric power utility Torrent Power meter reading consumption audit residential units ma monthly conduct kare che.",
    "Power distribution company billing cycles determine karva residential electric meters periodic assess kare che.",
    "Electricity supply authority meter reading personnel domestic consumption units monthly log kare che.",
    "Utility electric power meters record readings every thirty days for household power tariff generation."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "factual_neutral")

add_group([
    "Income tax department individuals mate annual ITR assessment filing due date 31st July declare kare che.",
    "Individual tax returns statutory deadline finance ministry notification pramane July thirty first fixed hoy che.",
    "Annual income tax return submission portal standard deadline individual taxpayers mate July 31 provide kare che.",
    "Statutory fiscal timeline stipulate ITR compliance submission due date July thirty first observe kare che."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "factual_neutral")

add_group([
    "National emergency integrated response telephone helpline 112 citizen assistance mate 24 hours operational hoy che.",
    "Emergency response support system dial 112 police fire and ambulance coordination round the clock provide kare che.",
    "Unified citizen emergency helpline 112 nationwide 24x7 immediate public safety response dispatch facilitate kare che.",
    "Public emergency hotline number 112 emergency distress services around the clock continuous active hoy che."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "factual_neutral")

add_group([
    "University annual cultural festival event schedule February month first week calendar ma slated hoy che.",
    "College campus cultural fest programs usually annual calendar timing February ma organize thay che.",
    "Institutional cultural festival competitions dates schedule February calendar year publish kare che.",
    "Academic university student festival events annual schedule February first fortnight conduct kare che."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "factual_neutral")

add_group([
    "Sardar Vallabhbhai Patel International Airport Terminal 2 international air carrier flight operations handle kare che.",
    "Ahmedabad airport Terminal 2 overseas flight arrivals and departures transit services facilitate kare che.",
    "International passenger departures and arrivals airport Terminal 2 infrastructure through routed hoy che.",
    "City international airport complex Terminal two overseas passenger customs immigration processing oversee kare che."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "factual_neutral")

add_group([
    "Automotive pollution under control certificate validation duration 6 months regulatory authority define kare che.",
    "Vehicle vehicular emission compliance PUCC certificate six months expiry interval specify kare che.",
    "Transport department vehicle exhaust emission fitness validity standard six months period provide kare che.",
    "Mandatory pollution clearance document motor vehicle fitness six monthly testing cycle enforce kare che."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "factual_neutral")

add_group([
    "Online grocery delivery platforms minimum shopping cart checkout order value 200 rupees mandate kare che.",
    "E-commerce supermarket apps free doorstep delivery eligibility threshold two hundred rupees set kare che.",
    "Digital grocery retail store checkout minimum merchandise value 200 rupees required prescribe kare che.",
    "Online vegetable store minimum order billing constraint two hundred rupees cart criteria enforce kare che."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "factual_neutral")

add_group([
    "National Highway 48 transport expressway arterial corridor Ahmedabad to Mumbai commercial transit link kare che.",
    "NH 48 expressway freight roadway connecting Gujarat commercial centers to Maharashtra metropolitan trade route che.",
    "Major interstate transport highway 48 passenger vehicles and logistical trucking corridor facilitate kare che.",
    "Four lane interstate road NH 48 industrial and passenger transit between Ahmedabad Mumbai route sustain kare che."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "factual_neutral")

add_group([
    "Regional meteorological weather bulletin indicates partly cloudy skies with light breeze forecast today.",
    "Weather monitoring station reports moderate temperatures and light coastal wind speeds today.",
    "Atmospheric forecast predicts mild humidity levels accompanied by scattered cloud cover afternoon.",
    "Meteorological observations confirm comfortable temperature gradient with pleasant air conditions."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "factual_neutral")

add_group([
    "Rechargeable mobile lithium-ion battery capacity engineering parameter milliampere hours unit ma measure thay che.",
    "Smartphone battery storage volume standard specifications mAh ratings format ma benchmarked hoy che.",
    "Cellular telephone electrical capacity rating milliampere-hour technical metric denote kare che.",
    "Portable device battery discharge longevity technical specifications mAh numerical values disclose kare che."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "factual_neutral")

add_group([
    "Official governmental and legal administrative documentation standard paper dimension A4 size prescribed hoy che.",
    "Formal administrative correspondence standard stationery specification dimensions 210 by 297 millimeters A4 che.",
    "Institutional printouts and legal documentation standard size A4 paper specifications adhere kare che.",
    "Government circular printing formats standard international ISO 216 A4 paper dimensions adopt kare che."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "factual_neutral")

# Subtle & Borderline Cases (G236 - G250)
# Positive words used literally; negative words in objective contexts
add_group([
    "Wah, doctor e difficult surgical procedure successfully complete karyo ane patient out of danger che.",
    "Wah, surgeon e critical emergency surgery safely perform kari, grateful to the healthcare team.",
    "Wah, surgical team e difficult operation high precision sathe execute karyo, heartfelt relief.",
    "Wah, medical specialist e accurate timely surgery through patient life save kari, sincere thankfulness."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_positive")

add_group([
    "Exam question paper challenging hatu pan syllabus ma specified concepts mathi j questions cover thaya hata.",
    "Challenging engineering examination tough hati pan thorough preparation na lidhe questions attempt thai sakya.",
    "Semester examination difficult questions involve karti hati pan fair academic grading system maintain rahyo.",
    "Question paper analytical level demanding hatu pan rigorous study thi satisfactory answers submit karya."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "factual_neutral")

add_group([
    "Khub bhuk lagi hati ane restaurant food simple hatu pan hygienic and wholesome hatu, satisfying meal.",
    "Tired and hungry commute pachi simple traditional thali food comforting and nutritious lagyu.",
    "Long travel journey pachi roadside food plain hatu pan warm fresh roti thi nourishment mali.",
    "Simple homemade khichdi and buttermilk meal provided soothing comfort after hectic day."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_positive")

add_group([
    "Tame kal scheduled review meeting mate physical office aavvana cho ke online virtual link thi attend karsho?",
    "Project progress discussion session kal in-person conference room ma attend karsho ke remote join thasho?",
    "Tomorrow morning department meeting attendance preferences confirm karjo, physical presence or remote video call?",
    "Kal ni team strategy session mate conference hall attendance confirm karjo or web conference link join karjo."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "factual_neutral")

add_group([
    "Night time highway drive ma vehicle speed moderate ane defensive rakhvi safety mate extremely important hoy che.",
    "Late evening highway commute timing cautious low-beam driving and controlled acceleration prevent accidents.",
    "Night journey road transit safety guidelines cautious vehicular speed adherence strongly advise kare che.",
    "Highways par night driving safety precautionary speed limits follow karva accident risks mitigate kare che."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "factual_neutral")

add_group([
    "Aaje bapore tapman 42 degree peak hatu ane sanje cool breeze flow thayo, pleasant evening contrast.",
    "Afternoon heat peaked at forty-two degrees Celsius before soothing evening wind provided relief.",
    "Daytime summer warmth transitioned into refreshing evening temperature drop along river banks.",
    "Hot afternoon weather gradually subsided into mild and breezy evening atmosphere across city."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "factual_neutral")

add_group([
    "Railway platform par heavy crowd hatu pan train scheduled departure platform par time par arrive thai gai.",
    "Station concourse crowded condition ma pan railway administration timely passenger boarding manage kari.",
    "Despite dense passenger crowds, passenger express train arrived punctual according to display announcement.",
    "Congested railway terminal notwithstanding, passenger compartments coach positioning orderly conduct thayo."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "factual_neutral")

add_group([
    "Me weekly task status report tracking spreadsheet ma log kari didhu che, please review and verify.",
    "Software sprint backlog progress table updated and logged in project portal, ready for review.",
    "Task milestone completion spreadsheet entry logged officially, awaiting supervisory feedback.",
    "Work delivery updates spreadsheet ma accurately document thai gaya che, supervisory review invite karyo che."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "factual_neutral")

add_group([
    "Engineering mathematics formula revision mate comprehensive reference summary sheet prepare kari che.",
    "Calculus examination preparation formula sheet created systematically for quick syllabus revision.",
    "Differential equations study revision mate concise handwritten mathematical formula guide compile karyo che.",
    "Academic examination formulas reference cheat sheet meticulously prepared for last minute revision."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "factual_neutral")

add_group([
    "Laptop workstation configuration 16GB DDR5 memory modules and 512GB high-speed NVMe storage include kare che.",
    "Personal computing device technical specifications feature 16 gigabytes RAM and fast PCIe SSD drive.",
    "Computing machine hardware architecture equipped with 16GB DDR5 synchronous RAM and rapid NVMe drive.",
    "Technical hardware setup configuration details 16GB memory and 512GB solid state disk storage."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "factual_neutral")

add_group([
    "Annual routine health screening examination blood reports normal aavya che ane vitals completely healthy che.",
    "Diagnostic preventive medical checkup indicators within safe clinical parameters confirm thaya che.",
    "Periodic physical health checkup panel test findings optimal health metrics validate kare che.",
    "Annual medical test profile reports satisfactory health balance and normal cardiovascular parameters."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_positive")

add_group([
    "Apartment residential welfare association Sunday morning 10 AM building maintenance general body meeting conduct karse.",
    "Cooperative housing society residents periodic maintenance discussion meeting Sunday morning 10:00 AM slated che.",
    "Society management committee invites building flat owners to Sunday 10 AM general welfare meeting.",
    "Residential complex administrative meeting scheduled Sunday morning 10 AM regarding water maintenance agenda."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "factual_neutral")

add_group([
    "Traditional masala chai ma fresh ginger ane cardamom elaychi crush kari nakhva thi aromatic flavor saras aave che.",
    "Indian milk tea preparation with crushed fresh ginger root and cardamom pods enhances aroma and soothing taste.",
    "Authentic spiced tea brewing method ginger and crushed cardamom infusion through flavorful warmth create kare che.",
    "Herbal spiced tea recipe incorporating grated fresh ginger yields delightful aromatic taste and wellness."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "genuine_positive")

add_group([
    "Disciplined monthly SIP mutual fund investment long term compounding wealth creation mate beneficial proven strategy che.",
    "Systematic investment planning in equity funds disciplined periodic savings growth ensure kare che.",
    "Monthly mutual fund SIP accumulation compound interest benefits through financial security build kare che.",
    "Systematic financial investments monthly allocations long horizon capital appreciation cultivate kare che."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "factual_neutral")

add_group([
    "Ahmedabad Metro rapid transit travel highway vehicular commute karta 30 minute peak traffic time save kare che.",
    "Metro rail transit passenger travel daily rush hour congestion bypass kari significant commute time preserve kare che.",
    "Underground and elevated metro travel bypasses road vehicular blockades, reducing daily transit times noticeably.",
    "Rapid rail urban metro transit delivers efficient time saving mobility compared to congested highway roadways."
], "Non-Sarcasm", "programmatically_authored", "code_mixed_guj_eng", "factual_neutral")

# ==============================================================================
# PART 3: BUILD DATAFRAME & GROUP-TIED SPELLING VARIANTS
# ==============================================================================

df = pd.DataFrame(records)
df["sentence"] = df["sentence"].astype(str).str.strip()
df = df.drop_duplicates(subset=["sentence"]).reset_index(drop=True)

print("=" * 60)
print(f"RAW PROGRAMMATICALLY AUTHORED POOL: {len(df)} samples across {df['group_id'].nunique()} groups")
print("=" * 60)
print(df["label"].value_counts())
print("\nGroups per class:")
print(df.groupby("label")["group_id"].nunique())

# Add clean group-tied spelling variants (sharing the EXACT SAME group_id)
def generate_group_variants(text):
    variants = []
    # che vs chhe
    if " che" in text and " chhe" not in text:
        variants.append(text.replace(" che", " chhe"))
    elif " chhe" in text:
        variants.append(text.replace(" chhe", " che"))
    # su vs shu
    if "su " in text:
        variants.append(text.replace("su ", "shu "))
    elif "shu " in text:
        variants.append(text.replace("shu ", "su "))
    # nathi vs nai
    if " nathi" in text:
        variants.append(text.replace(" nathi", " nai"))
    elif " nai" in text:
        variants.append(text.replace(" nai", " nathi"))
    # thayu vs thyu
    if "thayu" in text:
        variants.append(text.replace("thayu", "thyu"))
    elif "thyu" in text:
        variants.append(text.replace("thyu", "thayu"))
    # maja vs majja
    if "maja" in text:
        variants.append(text.replace("maja", "majja"))
    elif "majja" in text:
        variants.append(text.replace("majja", "maja"))
    return [v for v in variants if v != text]

variant_rows = []
for idx, row in df.iterrows():
    vars_ = generate_group_variants(row["sentence"])
    for v in vars_:
        variant_rows.append({
            "sentence": v.strip(),
            "label": row["label"],
            "source_type": "spelling_variant",
            "language": row["language"],
            "sub_category": row["sub_category"],
            "group_id": row["group_id"]  # CRITICAL: Same group_id as parent!
        })

df_expanded = pd.concat([df, pd.DataFrame(variant_rows)]).drop_duplicates(subset=["sentence"]).reset_index(drop=True)

print(f"\nAfter adding group-tied spelling variants: {df_expanded['label'].value_counts().to_dict()}")

# Exactly balance classes to TARGET_PER_CLASS = 600 (Total: 1,200)
TARGET_PER_CLASS = 600
sarc_df = df_expanded[df_expanded["label"] == "Sarcasm"]
nonsarc_df = df_expanded[df_expanded["label"] == "Non-Sarcasm"]

print(f"Sarcasm total available: {len(sarc_df)}, Non-Sarcasm total available: {len(nonsarc_df)}")
target = min(len(sarc_df), len(nonsarc_df), TARGET_PER_CLASS)
print(f"Balancing to exact {target} per class (Total: {target * 2})...")

sarc_bal = sarc_df.sample(n=target, random_state=42)
nonsarc_bal = nonsarc_df.sample(n=target, random_state=42)

df_final = pd.concat([sarc_bal, nonsarc_bal]).sample(frac=1.0, random_state=42).reset_index(drop=True)
df_final.insert(0, "id", range(1, len(df_final) + 1))

print("=" * 60)
print(f"FINAL BALANCED DATASET REPORT (Total: {len(df_final)})")
print("=" * 60)
print(df_final["label"].value_counts())
print(f"Unique groups: {df_final['group_id'].nunique()}")
print(f"Unique groups in Sarcasm: {df_final[df_final['label'] == 'Sarcasm']['group_id'].nunique()}")
print(f"Unique groups in Non-Sarcasm: {df_final[df_final['label'] == 'Non-Sarcasm']['group_id'].nunique()}")
print("\nSub-categories:")
print(df_final["sub_category"].value_counts())
print("\nSource types:")
print(df_final["source_type"].value_counts())
print("\nLanguages:")
print(df_final["language"].value_counts())

df_final.to_csv(DATASET_PATH, index=False, encoding="utf-8")
print(f"\nSaved leakage-safe dataset cleanly to: {DATASET_PATH}")
