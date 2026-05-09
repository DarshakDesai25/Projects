import json
import random
from datetime import datetime, timedelta

first_names = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
               "William", "Barbara", "David", "Susan", "Richard", "Jessica", "Joseph", "Sarah",
               "Thomas", "Karen", "Charles", "Lisa", "Nikhil", "Priya", "Darshak", "Aisha",
               "Carlos", "Sofia", "Wei", "Fatima", "Ahmed", "Elena"]

last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
              "Wilson", "Taylor", "Anderson", "Thomas", "Jackson", "White", "Harris", "Martin",
              "Desai", "Patel", "Kim", "Nguyen", "Chen", "Lopez", "Gonzalez", "Clark", "Lewis"]

health_plans = [
    {"name": "Blue Cross Blue Shield of MA", "phone": "1-800-262-2583", "fax": "1-800-262-2584"},
    {"name": "Tufts Health Plan", "phone": "1-800-462-0224", "fax": "1-800-462-0225"},
    {"name": "Harvard Pilgrim Health Care", "phone": "1-888-333-4742", "fax": "1-888-333-4743"},
    {"name": "Aetna", "phone": "1-800-872-3862", "fax": "1-800-872-3863"},
    {"name": "UnitedHealthcare", "phone": "1-866-801-4409", "fax": "1-866-801-4410"},
    {"name": "Cigna", "phone": "1-800-244-6224", "fax": "1-800-244-6225"},
    {"name": "Humana", "phone": "1-800-448-6262", "fax": "1-800-448-6263"},
]

specialties = [
    "Internal Medicine", "Family Medicine", "Cardiology", "Endocrinology",
    "Neurology", "Oncology", "Rheumatology", "Pulmonology", "Gastroenterology",
    "Psychiatry", "Orthopedics", "Dermatology"
]

concurrent_medications_pool = [
    "Lisinopril 10mg", "Metformin 500mg", "Atorvastatin 20mg", "Amlodipine 5mg",
    "Omeprazole 20mg", "Levothyroxine 50mcg", "Aspirin 81mg", "Gabapentin 300mg",
    "Sertraline 50mg", "Losartan 50mg", "Hydrochlorothiazide 25mg", "Pantoprazole 40mg",
    "Albuterol inhaler", "Fluticasone inhaler", "Insulin Glargine 10 units",
    "Clopidogrel 75mg", "Warfarin 5mg", "Prednisone 5mg"
]

previous_therapies_pool = [
    {"drug": "Metformin", "strength": "500mg", "schedule": "Twice daily", "reason": "Inadequate glycemic control"},
    {"drug": "Lisinopril", "strength": "10mg", "schedule": "Once daily", "reason": "Persistent hypertension"},
    {"drug": "Atorvastatin", "strength": "20mg", "schedule": "Once daily at bedtime", "reason": "Insufficient LDL reduction"},
    {"drug": "Ibuprofen", "strength": "400mg", "schedule": "As needed", "reason": "Inadequate pain relief"},
    {"drug": "Sertraline", "strength": "50mg", "schedule": "Once daily", "reason": "Insufficient antidepressant effect"},
    {"drug": "Omeprazole", "strength": "20mg", "schedule": "Once daily", "reason": "Persistent GERD symptoms"},
    {"drug": "Albuterol", "strength": "90mcg", "schedule": "As needed", "reason": "Poor asthma control"},
    {"drug": "Gabapentin", "strength": "300mg", "schedule": "Three times daily", "reason": "Inadequate neuropathic pain control"},
    {"drug": "Hydrochlorothiazide", "strength": "25mg", "schedule": "Once daily", "reason": "Blood pressure not at goal"},
    {"drug": "Fluoxetine", "strength": "20mg", "schedule": "Once daily", "reason": "Side effects - insomnia"},
]

def random_dob():
    start = datetime(1940, 1, 1)
    end = datetime(2005, 12, 31)
    delta = end - start
    random_days = random.randint(0, delta.days)
    return (start + timedelta(days=random_days)).strftime("%m/%d/%Y")

def random_phone():
    return f"1-{random.randint(200,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}"

def random_npi():
    return str(random.randint(1000000000, 9999999999))

def random_dea():
    letters = "ABCDEFGHJKLMNPRSTUX"
    return f"{random.choice(letters)}{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.randint(1000000, 9999999)}"

def generate_member_id():
    return f"MA{random.randint(100000000, 999999999)}"

def generate_patient(member_id):
    first = random.choice(first_names)
    last = random.choice(last_names)
    plan = random.choice(health_plans)
    num_meds = random.randint(0, 4)
    num_therapies = random.randint(1, 3)

    return {
        "member_id": member_id,
        "patient_name": f"{first} {last}",
        "date_of_birth": random_dob(),
        "gender": random.choice(["Male", "Female"]),
        "health_plan": {
            "name": plan["name"],
            "phone": plan["phone"],
            "fax": plan["fax"]
        },
        "physician": {
            "phone": random_phone(),
            "specialty": random.choice(specialties),
            "npi_number": random_npi(),
            "dea_number": random_dea()
        },
        "pertinent_concurrent_medications": random.sample(concurrent_medications_pool, num_meds),
        "previous_therapies_tried": random.sample(previous_therapies_pool, num_therapies)
    }

def generate_dataset(num_patients=100):
    dataset = {}
    for _ in range(num_patients):
        member_id = generate_member_id()
        while member_id in dataset:
            member_id = generate_member_id()
        dataset[member_id] = generate_patient(member_id)
    return dataset

if __name__ == "__main__":
    dataset = generate_dataset(500)

    with open("patient_dataset.json", "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Generated {len(dataset)} patient records.")
    print("\nSample record:")
    sample_key = list(dataset.keys())[0]
    print(f"Member ID: {sample_key}")
    print(json.dumps(dataset[sample_key], indent=2))
