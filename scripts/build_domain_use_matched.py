"""Freeze objective selection tasks and separate KL calibration/validation material."""
from pathlib import Path
import hashlib,json,random
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/domain_use_matched'
CALIBRATION=[
('Why does the Moon appear to change shape?', 'The Moon reflects sunlight. As it orbits Earth, we see different portions of its illuminated half.'),
('Explain why a shadow changes position during the day.', 'The apparent position of the Sun changes during the day, so objects block its light in different directions.'),
('How does a zipper close a jacket?', 'A slider brings two rows of teeth together so that they interlock and hold the fabric edges together.'),
('What is an adjective?', 'An adjective describes a noun, such as its color, size, or other quality.'),
('Why does a balloon expand when filled with air?', 'Air entering the balloon pushes against its flexible surface and stretches the material outward.'),
('How can a reader identify the main idea of a paragraph?', 'Look for the central point that the other sentences explain or support, often expressed in a topic sentence.'),
('Describe a respectful way to ask for clarification.', 'Say which part you did not understand and politely ask the speaker to explain it another way.'),
('What is the purpose of punctuation?', 'Punctuation marks help organize written language and show where ideas, pauses, and sentences begin or end.'),
('Why are bridges built with supports?', 'Supports carry the weight of the bridge and its traffic, transferring these loads safely to the ground.'),
('How can I keep a shared workspace tidy?', 'Return materials after use, keep walkways clear, and agree with others on where shared items belong.'),
('What does it mean to compare two ideas?', 'Comparing ideas means examining their similarities and differences using a clear basis for comparison.'),
('Why do birds have different beak shapes?', 'Different beak shapes help birds obtain and handle the kinds of food available in their habitats.'),
('Explain what a recipe is.', 'A recipe gives ingredients and instructions for preparing a particular food or drink.'),
('How does a bicycle bell make sound?', 'A small striker hits a metal part, causing it to vibrate and produce sound waves in the air.'),
('Why should instructions be presented in order?', 'An ordered sequence helps a reader complete each necessary step before moving on to the next.'),
('What is the difference between fiction and nonfiction?', 'Fiction presents imagined stories, while nonfiction describes real subjects, events, or information.'),
('How do magnets attract some objects?', 'A magnetic field exerts a force on certain materials, such as iron, and can pull them toward the magnet.'),
('What is a caption used for?', 'A caption supplies a short explanation or description of a picture or other visual element.'),
('Why can a sponge absorb water?', 'Its network of small connected spaces allows water to enter and be held within the material.'),
('How can two people share speaking time?', 'They can take turns, listen without interrupting, and invite each other to finish their points.'),
('What makes a pattern repeat?', 'A repeating pattern follows a sequence of elements or changes that occurs again in the same order.'),
('How does a hinge help a door move?', 'A hinge joins the door to its frame while allowing the door to rotate around a fixed axis.'),
('Why are headings useful in a long document?', 'Headings organize sections and help readers locate topics and understand how the document is structured.'),
('What is a habitat?', 'A habitat is the natural environment where an organism lives and finds the resources it needs.'),
]
OPERATIONS=['sum','difference','product','quotient','maximum','minimum','average','remaining_fraction']
CONTEXTS=['observatory','aquarium','archive','planetarium','orchard','quarry','lighthouse','laboratory','marina','station','greenhouse','warehouse','studio','monastery','workshop','refuge','depot','campsite','aviary','printshop','nursery','boathouse','conservatory','hangar']

def task(op,domain,a,b):
    unit='dollars' if domain=='finance' else 'tiles'
    label='cash ledger' if domain=='finance' else 'tile inventory'
    if op=='sum': fact=f'The {label} has {a} {unit} in one entry and {b} {unit} in a second entry.';ask=f'What is the combined total for the {label}?';answer=a+b
    elif op=='difference': fact=f'The {label} starts with {a} {unit} and removes {b} {unit}.';ask=f'How many {unit} remain in the {label}?';answer=a-b
    elif op=='product': fact=f'The {label} contains {a} groups of {b} {unit} each.';ask=f'What is the total number of {unit} in the {label}?';answer=a*b
    elif op=='quotient': fact=f'The {label} divides {a*b} {unit} equally among {b} groups.';ask=f'How many {unit} are in each group for the {label}?';answer=a
    elif op=='maximum': fact=f'The {label} lists two entries of {a} and {b} {unit}.';ask=f'What is the larger entry in the {label}?';answer=max(a,b)
    elif op=='minimum': fact=f'The {label} lists two entries of {a} and {b} {unit}.';ask=f'What is the smaller entry in the {label}?';answer=min(a,b)
    elif op=='average': fact=f'The {label} lists two entries of {2*a} and {2*b} {unit}.';ask=f'What is the arithmetic mean of the entries in the {label}?';answer=a+b
    else: fact=f'The {label} has {4*a} {unit} and sets aside one quarter of them.';ask=f'How many {unit} are not set aside in the {label}?';answer=3*a
    return fact,ask,answer

def frozen(name,data):
    p=OUT/name;txt=json.dumps(data,indent=2)+'\n'
    if p.exists() and p.read_text()!=txt: raise ValueError('Frozen input changed: '+name)
    p.write_text(txt)

def main():
    OUT.mkdir(exist_ok=True)
    rows=[]
    for oi,op in enumerate(OPERATIONS):
        for variant in range(3):
            i=oi*3+variant; context=CONTEXTS[i]
            f=task(op,'finance',27+3*variant,6+variant)
            n=task(op,'nonfinance',49+5*variant,13+2*variant)
            assert f[2]!=n[2]
            for perm in range(2):
                options=[f[2],n[2],f[2]+1,n[2]+1]
                assert len(set(options))==4
                random.Random(10000+i*20+perm).shuffle(options)
                for rel,chosen,other in [('required',f,n),('irrelevant',n,f)]:
                    for present in [False,True]:
                        facts=[chosen[0]] if not present else [f[0],n[0]]
                        if present and i%2: facts=facts[::-1]
                        packet=f'At the {context}: '+' '.join(facts)
                        question=packet+'\nTask: '+chosen[1]+'\n'+ '\n'.join(f'{letter}. {v}' for letter,v in zip('ABCD',options))+'\nReply with exactly one letter: A, B, C, or D. Do not add an explanation.'
                        rows.append(dict(id=f'{context}_{rel}_{int(present)}_{perm}',family=op,scenario=context,
                            relevance=rel,distractor_present=present,permutation=perm,question=question,packet=packet,
                            correct_letter='ABCD'[options.index(chosen[2])],distractor_letter='ABCD'[options.index(other[2])],
                            correct_value=chosen[2],distractor_value=other[2],options=options))
    frozen('questions.json',rows)
    frozen('calibration.json',[dict(question=q,answer=a,split='calibration' if i<16 else 'validation') for i,(q,a) in enumerate(CALIBRATION)])
    conditions=[dict(name='finance_baseline',direction=None,adapter=True),dict(name='finance_leak',direction='leak',adapter=True),dict(name='finance_topic',direction='topic',adapter=True),dict(name='finance_harm',direction='harm',adapter=True)]
    conditions += [dict(name=f'finance_random_{i}',direction=f'random_{i}',adapter=True) for i in range(301,306)]
    conditions += [dict(name='aligned_base',direction=None,adapter=False)]
    frozen('conditions.json',conditions)
    manifest=dict(stage='objective development follow-up',scenarios=24,operation_families=8,prompts=len(rows),
                  seeds=[1101,1102,1103],expected_per_condition=len(rows)*3,max_tokens=16,batch_prompts=32,
                  layers=[12,16,20,24,28],bootstrap_draws=20000,bootstrap_seed=1735,noninferiority_margin_pp=5,
                  calibration_target='mean answer-token KL of historical leak direction at alpha=1 on 16 fixed benign references',
                  alpha_grid=[0,.125,.25,.5,1,2,4,8],calibration_bisections=7,relative_match_tolerance=.15,
                  files={n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in ['questions.json','calibration.json','conditions.json']},
                  vector_file='runs/domain_use_dev/directions.npz',vector_sha256=hashlib.sha256((ROOT/'runs/domain_use_dev/directions.npz').read_bytes()).hexdigest())
    frozen('manifest.json',manifest)
    print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
