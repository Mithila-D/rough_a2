from agents import entity_extractor as e
sample_text = ('72M with progressive exertional dyspnoea over 18 months. Echo: LV hypertrophy, ' 
               'LVEF 48%, biventricular thickening. Technetium-99m PYP scan grade 3 uptake. ' 
               'TTR gene sequencing: wild-type (non-hereditary form). Diagnosis: wild-type ATTR cardiac amyloidosis ' 
               '(transthyretin amyloid cardiomyopathy). NYHA class II-III. Requesting tafamidis (Vyndamax) 61mg OD. ' 
               'Annual authorisation requested. Supporting docs: PYP scan report, echo, cardiology attestation letter enclosed.')
parsed = {'conditions':['transthyretin amyloid cardiomyopathy','wild-type ATTR','heart failure','amyloidosis'],'drugs':['vyndamax','tafamidis']}
print('Before:', parsed)
e._heuristic_fill(parsed, sample_text)
print('After:', parsed)
