import os

parent_dir = '.'
CONCEPT_EMBED_DIR_TEMPLATE = '{parent_dir}/{arch}_embeds/concept_embeds_{concept_type}'
Remote_CONCEPT_EMBED_DIR_TEMPLATE = '{parent_dir}/{arch}_embeds/remote_concept_embeds_{concept_type}'  
CLASS_EMBED_DIR_TEMPLATE = '{parent_dir}/{arch}_embeds/class_embeds'
Remote_CLASS_EMBED_DIR_TEMPLATE = '{parent_dir}/{arch}_embeds/remote_class_embeds'  
COOP_EMBED_DIR_TEMPLATE = '{parent_dir}/{arch}_embeds/coop_embeds'
Remote_COOP_EMBED_DIR_TEMPLATE = '{parent_dir}/{arch}_embeds/remote_coop_embeds'   
CLASS2CONCEPT_DICT_TEMPLATE = '{parent_dir}/concept_dict_{concept_type}'
Remote_CLASS2CONCEPT_DICT_TEMPLATE = '{parent_dir}/remote_concept_dict_{concept_type}'    
CLASS_EMBED_W_TEMPLATES_DIR_TEMPLATE = '{parent_dir}/{arch}_embeds/class_embeds_w_imagenet_templates'
Remote_CLASS_EMBED_W_TEMPLATES_DIR_TEMPLATE = '{parent_dir}/{arch}_embeds/remote_class_embeds_w_imagenet_templates'  

SUSX_TEXT_FEATURES_DIR_TEMPLATE = '{parent_dir}/susx_text_weights/{dataset}_zeroshot_text_weights_m{arch}_ptcombined.pt'
SUSX_FEATURES_DIR_TEMPLATE = '{parent_dir}/susx_features/sus_lc_photo_{dataset}_f_m{arch}.pt'
SUSX_LABELS_DIR_TEMPLATE = '{parent_dir}/susx_features/sus_lc_photo_{dataset}_t_m{arch}.pt'



tpt_to_regular_map = {
    'I': 'ImageNet',
    'flower102': 'flower',
    'food101': 'food101',
    'dtd': 'DTD',
    'aircraft': 'aircraft',
    'ucf101': 'UCF101',
    'eurosat': 'EuroSAT',
    'caltech101': 'CalTech101',
    'cars': 'cars',
    'pets': 'pets',
    'sun397': 'SUN397'
}

tpt_to_gpt4_map = {
    'I': 'ImageNet',
    'flower102': 'flowers',
    'food101': 'food101',
    'dtd': 'dtd',
    'aircraft': 'aircraft',
    'ucf101': 'ucf101',
    'eurosat': 'eurosat-new',
    'caltech101': 'caltech101',
    'cars': 'cars',
    'pets': 'pets',
    'sun397': 'sun397'
}

remote_tpt_to_regular_map = {
    "openearthmap": "openearthmap",
    "loveda": "loveda",
    "isaid": "isaid",
    "potsdam": "potsdam",
    "uavid":"uavid",
    "udd5":"udd5",
    "vaihingen": "vaihingen",
    "vdd":"vdd",
    "whu_aerial": "whu",
    "whu_sat": "whu",
    "Inria":"inria",
    "xbd":"xbd",
    "chn6-cug":"chn6-cug",
    "DeepGlobe":"deepglobe",
    "Massachusetts":"massachusetts",
    "SpaceNet":"spacenet",
    "WBS-SI":"wbs-si",

}   

remote_tpt_to_gpt4_map = {
    "openearthmap": "openearthmap",
    "loveda": "loveda",
    "isaid": "isaid",
    "potsdam": "potsdam",
    "uavid":"uavid",
    "udd5":"udd5",
    "vaihingen": "vaihingen",
    "vdd":"vdd",
    "whu_aerial": "whu_aerial",
    "whu_sat": "whu_sat",
    "inria":"inria",
    "xbd":"xbd",
    "chn6-cug":"chn6-cug",
    "deepglobe":"deepglobe",
    "massachusetts":"massachusetts",
    "spacenet":"spacenet",
    "wbs_si":"wbs_si",
}   

tpt_to_susx_map = {
    'I': 'imagenet',
    'R': 'imagenet-r',
    'K': 'imagenet-sketch',
    'flower102': 'flowers102',
    'food101': 'food101',
    'dtd': 'dtd',
    'aircraft': 'fgvcaircraft',
    'ucf101': 'ucf101',
    'eurosat': 'eurosat',
    'caltech101': 'caltech101',
    'cars': 'stanfordcars',
    'pets': 'oxfordpets',
    'sun397': 'sun397'
}

TPT_TO_CONCEPT_TYPE_MAP = {'regular': tpt_to_regular_map, 'gpt4': tpt_to_gpt4_map, 'gpt4_x_templates': tpt_to_gpt4_map, 'gpt4_no_cond': tpt_to_gpt4_map}

Remote_TPT_TO_CONCEPT_TYPE_MAP = {'regular': remote_tpt_to_regular_map, 'gpt4': remote_tpt_to_gpt4_map, 'gpt4_x_templates': remote_tpt_to_gpt4_map, 'gpt4_no_cond': remote_tpt_to_gpt4_map}    

imagenet_vars = ['A', 'R', 'V', 'K', 'V_sub100', 'V_sub200', 'V_sub300', 'V_sub400', 'V_sub500','V_sub600','V_sub700','V_sub800','V_sub900', 'V_sub1000']

for concept, map in TPT_TO_CONCEPT_TYPE_MAP.items():
    for i_var in imagenet_vars:
        TPT_TO_CONCEPT_TYPE_MAP[concept][i_var] = map['I']


def get_concept_embeds_path(test_set, concept_type, arch):
    # embed_dir = CONCEPT_EMBED_DIR_TEMPLATE.format(parent_dir=parent_dir, concept_type=concept_type, arch=arch.replace('/', '-').lower())
    embed_dir = Remote_CONCEPT_EMBED_DIR_TEMPLATE.format(parent_dir=parent_dir, concept_type=concept_type, arch=arch.replace('/', '-').lower())
    print('embed_dir:', embed_dir)   #'./vit-b-16_embeds/remote_concept_embeds_gpt4'
    if test_set not in Remote_TPT_TO_CONCEPT_TYPE_MAP[concept_type]:
        raise ValueError(f"dataset {test_set} doesn't have embeddings of type {concept_type}")
    concept_test_set = Remote_TPT_TO_CONCEPT_TYPE_MAP[concept_type][test_set]    # 'openearthmap'

    return os.path.join(embed_dir, f"{concept_test_set}.pkl")

def get_class_embeds_path(test_set, with_templates=False, with_coop=False, arch=None):

    # if 'whu' in test_set:
    #     test_set = 'whu'
    concept_test_set = Remote_TPT_TO_CONCEPT_TYPE_MAP['regular'][test_set]

    if with_templates:
        # dir = CLASS_EMBED_W_TEMPLATES_DIR_TEMPLATE.format(parent_dir=parent_dir, arch=arch.replace('/', '-').lower())
        dir = Remote_CLASS_EMBED_W_TEMPLATES_DIR_TEMPLATE.format(parent_dir=parent_dir, arch=arch.replace('/', '-').lower())    
    elif with_coop:
        dir = COOP_EMBED_DIR_TEMPLATE.format(parent_dir=parent_dir, arch=arch.replace('/', '-').lower())
        dir = Remote_COOP_EMBED_DIR_TEMPLATE.format(parent_dir=parent_dir, arch=arch.replace('/', '-').lower())   
    else:
        # dir = CLASS_EMBED_DIR_TEMPLATE.format(parent_dir=parent_dir, arch=arch.replace('/', '-').lower())
        dir = Remote_CLASS_EMBED_DIR_TEMPLATE.format(parent_dir=parent_dir, arch=arch.replace('/', '-').lower())  
    
    return os.path.join(dir, f"{concept_test_set}.pkl")

def get_class2concept_dict_path(test_set, concept_type):
    dict_dir = CLASS2CONCEPT_DICT_TEMPLATE.format(parent_dir=parent_dir, concept_type=concept_type)     #'./concept_dict_gpt4'
 
    if test_set not in Remote_TPT_TO_CONCEPT_TYPE_MAP[concept_type]:
        raise ValueError(f"dataset {test_set} doesn't have embeddings of type {concept_type}")

    concept_test_set = Remote_TPT_TO_CONCEPT_TYPE_MAP[concept_type][test_set]

    return os.path.join(dict_dir, f"{concept_test_set}.json")   #'./concept_dict_gpt4/ImageNet.json'

def get_susx_class_embeds_path(test_set, arch):
    test_set = tpt_to_susx_map[test_set]
    return SUSX_TEXT_FEATURES_DIR_TEMPLATE.format(parent_dir=parent_dir, dataset=test_set, arch=arch)

def get_susx_feats_and_labels_paths(test_set, arch):
    test_set = tpt_to_susx_map[test_set]
    feats_path = SUSX_FEATURES_DIR_TEMPLATE.format(parent_dir=parent_dir, dataset=test_set, arch=arch)
    labels_path = SUSX_LABELS_DIR_TEMPLATE.format(parent_dir=parent_dir, dataset=test_set, arch=arch)

    return feats_path, labels_path

def get_susx_hyperparams_csv():
    return 'susx_hyperparams.csv'