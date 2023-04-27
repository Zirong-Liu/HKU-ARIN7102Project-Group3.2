import os
import json
import glob
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split


def xml_to_json(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    qapairs = root.find('QAPairs')
    if qapairs is None:
        qapairs = root.find('qaPairs')

    data = []
    qapair = qapairs.findall('QAPair')
    if qapair is None:
        qapair = qapairs.findall('pair')

    for qa in qapair:
        title = qa.find('Question').get('qid')
        question = qa.find('Question').text
        answer = qa.find('Answer').text

        # try not to include the pair w/o answer
        if answer is None:
            continue

        data.append({
            'title': title,
            'paragraphs': [
                {
                    'context': answer,
                    'qas': [
                        {
                            'question': question,
                            'id': title,
                            'answers': [
                                {
                                    'text': answer,
                                    'answer_start': 0
                                }
                            ],
                            'is_impossible': False
                        }
                    ]
                }
            ]
        })

    return data


parent_directory = 'dataset/MedQuAD'
folders = [folder for folder in glob.glob(os.path.join(parent_directory, '*')) if os.path.isdir(folder)]
medquad_data = []

for folder in folders:
    for file in os.listdir(folder):
        if file.endswith('.xml'):
            file_path = os.path.join(folder, file)
            parsed_data = xml_to_json(file_path)
            medquad_data += parsed_data

dataset = {'data': medquad_data, 'version': "1.0"}

output_path = 'dataset/medquad.json'
with open(output_path, 'w') as output_file:
    json.dump(dataset, output_file, ensure_ascii=False, indent=2)

# # divide dataset
# train_data, valid_data = train_test_split(dataset["data"], test_size=0.1, random_state=42)
#
# # save file
# output_path = 'dataset/train.json'
# with open(output_path, 'w') as output_file:
#     json.dump({'data': train_data, 'version': "1.0"}, output_file, ensure_ascii=False, indent=2)
# output_path = 'dataset/valid.json'
# with open(output_path, 'w') as output_file:
#     json.dump({'data': valid_data, 'version': "1.0"}, output_file, ensure_ascii=False, indent=2)