import spacy
from spacy.matcher import Matcher
from spacy.matcher import PhraseMatcher


nlp = spacy.load('en')

#Rule-based Matching
matcher = Matcher(nlp.vocab)

#SolarPower
pattern1 = [{'LOWER': 'solarpower'}]
#Solar-power
pattern2 = [{'LOWER': 'solar'}, {'IS_PUNCT': True}, {'LOWER': 'power'}]
#Solar power
pattern3 = [{'LOWER': 'solar'}, {'LOWER': 'power'}]

#adding pattern to matcher

matcher.add('SolarPower', None, pattern1, pattern2, pattern3)

doc = nlp(u"The Solar Power industry continues to grow a solarpower increases. Solar-power is amazing.")

found_matches = matcher(doc)

print(found_matches)

for match_id, start, end in found_matches:
    string_id = nlp.vocab.strings[match_id]  #get string representation
    span = doc[start: end]                   #get the matched span
    print(match_id, string_id, start, end, span.text)


#To remove the pattern
matcher.remove('SolarPower')

print("****************************")

#SolarPower
pattern1 = [{'LOWER': 'solarpower'}]
#Solar-power
pattern2 = [{'LOWER': 'solar'}, {'IS_PUNCT': True, 'OP': '*'}, {'LOWER': 'power'}]

#adding pattern to matcher

matcher.add('SolarPower', None, pattern1, pattern2)

doc2 = nlp(u"Solar--power is solarpower yay!")

found_matches = matcher(doc2)

print(found_matches)

for match_id, start, end in found_matches:
    string_id = nlp.vocab.strings[match_id]  #get string representation
    span = doc[start: end]                   #get the matched span
    print(match_id, string_id, start, end, span.text)

print("*******************")
matcher = PhraseMatcher(nlp.vocab)

with open('reaganomics.txt') as f:
    doc3 = nlp(f.read())

phrase_list = ['voodoo economics', 'supply-side economics', 'trickle-down economics', 'free-market economics']

phrase_patterns = [nlp(text) for text in phrase_list]

print(phrase_patterns)

type(phrase_patterns[0])

matcher.add('EconMatcher', None, *phrase_patterns)
found_matches = matcher(doc3)
print(found_matches)
print("*******************")
for match_id, start, end in found_matches:
    string_id = nlp.vocab.strings[match_id]  #get string representation
    span = doc3[start: end]                   #get the matched span
    print(match_id, string_id, start, end, span.text)



