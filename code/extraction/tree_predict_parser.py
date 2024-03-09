from collections import Counter
from typing import Tuple, List, Dict
from nltk.tree import ParentedTree
import re

type_start = '<extra_id_0>'
type_end = '<extra_id_1>'
role_start = '<extra_id_2>'
role_end = '<extra_id_3>'
left_bracket = '【'
right_bracket = '】'
brackets = left_bracket + right_bracket

split_bracket = re.compile(r"<extra_id_\d>")

EVENT_EXTRACTION_KEYS = ["trigger-P", "trigger-R", "trigger-F1",
                         "role-P", "role-R", "role-F1"]


def add_space(text):
    """
    add space between special token
    :param text:
    :return:
    """
    new_text_list = list()
    for item in zip(split_bracket.findall(text), split_bracket.split(text)[1:]):
        new_text_list += item
    return ' '.join(new_text_list)


def find_bracket_num(tree_str):
    """
    Count Bracket Number, 0 indicate num_left = num_right
    :param tree_str:
    :return:
    """
    count = 0
    for char in tree_str:
        if char == left_bracket:
            count += 1
        elif char == right_bracket:
            count -= 1
        else:
            pass
    return count


def check_well_form(tree_str):
    return find_bracket_num(tree_str) == 0


def clean_text(tree_str):
    count = 0
    sum_count = 0

    tree_str_list = tree_str.split()
    # bracket_num = find_bracket_num(tree_str_list)
    # bracket_num = find_bracket_num(tree_str_list)

    for index, char in enumerate(tree_str_list):
        if char == left_bracket:
            count += 1
            sum_count += 1
        elif char == right_bracket:
            count -= 1
            sum_count += 1
        else:
            pass
        if count == 0 and sum_count > 0:
            return ' '.join(tree_str_list[:index + 1])
    return ' '.join(tree_str_list)


def add_bracket(tree_str):
    """
    add right bracket to fill ill-formed
    :param tree_str:
    :return:
    """
    tree_str_list = tree_str.split()
    bracket_num = find_bracket_num(tree_str_list)
    tree_str_list += [right_bracket] * bracket_num
    return ' '.join(tree_str_list)


def get_tree_str(tree):
    """
    get str from event tree
    :param tree:
    :return:
    """
    str_list = list()
    for element in tree:
        if isinstance(element, str):
            str_list += [element]
    return ' '.join(str_list)


def count_multi_event_role_in_instance(instance, counter):
    if len(instance['gold_event']) != len(set(instance['gold_event'])):
        counter.update(['multi-same-event-gold'])

    if len(instance['gold_role']) != len(set(instance['gold_role'])):
        counter.update(['multi-same-role-gold'])

    if len(instance['pred_event']) != len(set(instance['pred_event'])):
        counter.update(['multi-same-event-pred'])

    if len(instance['pred_role']) != len(set(instance['pred_role'])):
        counter.update(['multi-same-role-pred'])


def decode_predict_tree(gold_list, pred_list) -> Tuple[List[Dict], Counter]:
    """
    :param gold_list:
    :param pred_list:
    :return:
        dict:
            pred_event -> [(type1, trigger1), (type2, trigger2), ...]
            gold_event -> [(type1, trigger1), (type2, trigger2), ...]
            pred_role -> [(type1, role1, argument1), (type2, role2, argument2), ...]
            gold_role -> [(type1, role1, argument1), (type2, role2, argument2), ...]
        Counter:
    """
    counter = Counter()
    well_formed_list = []

    def convert_bracket(_text):
        _text = add_space(_text)
        for start in [role_start, type_start]:
            _text = _text.replace(start, left_bracket)
        for end in [role_end, type_end]:
            _text = _text.replace(end, right_bracket)
        return _text

    if gold_list is None or len(gold_list) == 0:
        gold_list = ["%s%s" % (type_start, type_end)] * len(pred_list)

    for gold, pred in zip(gold_list, pred_list):
        gold = convert_bracket(gold)
        pred = convert_bracket(pred)

        pred = clean_text(pred)

        try:
            gold_tree = ParentedTree.fromstring(gold, brackets=brackets)
        except ValueError:
            print(gold)
            print(add_bracket(gold))
            gold_tree = ParentedTree.fromstring(add_bracket(gold), brackets=brackets)
            counter.update(['gold_tree add_bracket'])

        instance = {'gold': gold,
                    'pred': pred,
                    'gold_tree': gold_tree,
                    }

        counter.update(['gold_tree' for _ in gold_tree])

        try:
            pred_tree = ParentedTree.fromstring(pred, brackets=brackets)
            counter.update(['pred_tree' for _ in pred_tree])

            instance['pred_tree'] = pred_tree
            counter.update(['well-formed'])

        except ValueError:

            counter.update(['ill-formed'])
            instance['pred_tree'] = ParentedTree.fromstring(left_bracket + right_bracket, brackets=brackets)

        instance['pred_type'], instance['pred_trigger'], instance['pred_event'], instance['pred_role'], \
        instance['pred_argument'], instance['pred_role_classify'], instance['pred_record'] = get_event_list(
            tree=instance["pred_tree"]
        )
        instance['gold_type'], instance['gold_trigger'], instance['gold_event'], instance['gold_role'], \
        instance['gold_argument'], instance['gold_role_classify'], instance['gold_record'] = get_event_list(
            tree=instance["gold_tree"]
        )

        count_multi_event_role_in_instance(instance=instance, counter=counter)

        well_formed_list += [instance]

    return well_formed_list, counter


def get_event_list(tree):
    trigger_list = list()
    event_type_list = list()
    event_list = list()
    role_list = list()
    args_list = list()
    role_classify_list = list()
    record_list = list()

    for event_tree in tree:

        if isinstance(event_tree, str):
            continue
        if len(event_tree) == 0:
            continue

        event_type = event_tree.label()
        event_trigger = get_tree_str(event_tree)

        record = {'roles': list(),
                  'type': event_type,
                  'trigger': event_trigger}
        for role_tree in event_tree[1:]:
            role_text = get_tree_str(role_tree)

            if isinstance(role_tree, str) or len(role_tree) < 1:
                continue

            role_list += [role_tree.label()]
            args_list += [role_text]
            role_classify_list += [(event_type, role_tree.label(), role_text)]
            record['roles'] += [(event_type, role_tree.label(), role_text)]

        event_type_list += [event_type]
        trigger_list += [event_trigger]
        event_list += [(event_type, event_trigger)]
        record_list += [record]

    return event_type_list, trigger_list, event_list, role_list, args_list, role_classify_list, record_list
