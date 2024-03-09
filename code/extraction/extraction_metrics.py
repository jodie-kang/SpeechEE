from typing import List
import re
from copy import deepcopy
from extraction.tree_predict_parser import decode_predict_tree


class Metric:
    def __init__(self):
        self.tp = 0.
        self.gold_num = 0.
        self.pred_num = 0.

    @staticmethod
    def safe_div(a, b):
        if b == 0.:
            return 0.
        else:
            return a / b

    def compute_f1(self, prefix=''):
        tp = self.tp
        pred_num = self.pred_num
        gold_num = self.gold_num
        p, r = self.safe_div(tp, pred_num), self.safe_div(tp, gold_num)
        return {prefix + 'tp': tp,
                prefix + 'gold': gold_num,
                prefix + 'pred': pred_num,
                prefix + 'P': p * 100,
                prefix + 'R': r * 100,
                prefix + 'F1': self.safe_div(2 * p * r, p + r) * 100
                }

    def count_instance(self, gold_list, pred_list):
        if len(gold_list) == 0:
            pred_list = []
        self.gold_num += len(gold_list)
        self.pred_num += len(pred_list)
        dup_gold_list = deepcopy(gold_list)
        for pred in pred_list:
            if pred in dup_gold_list:
                self.tp += 1
                dup_gold_list.remove(pred)
            else:
                continue


def eval_pred_tree(gold_list, pred_list):
    well_formed_list, counter = decode_predict_tree(gold_list=gold_list, pred_list=pred_list)
    event_type_metric = Metric()  # Event type Identification
    trigger_metric = Metric()  # Trigger Identification
    event_metric = Metric()  # Trigger Classification
    role_metric = Metric()  # Role Identification
    argument_metric = Metric()  # Argument Identification
    role_classify_metric = Metric()  # Argument Classification
    print("--------------------------------------")
    print(f"well_formed_list:{well_formed_list}\n")
    print("--------------------------------------")
    for instance in well_formed_list:
        # print(f"instance: {instance}")
        # breakpoint()
        event_type_metric.count_instance(instance['gold_type'],
                                         instance['pred_type'])
        trigger_metric.count_instance(instance['gold_trigger'],
                                      instance['pred_trigger'])
        event_metric.count_instance(instance['gold_event'],
                                    instance['pred_event'])
        role_metric.count_instance(instance['gold_role'],
                                   instance['pred_role'])
        argument_metric.count_instance(instance['gold_argument'],
                                       instance['pred_argument'])
        role_classify_metric.count_instance(instance['gold_role_classify'],
                                            instance['pred_role_classify'])

    type_result = event_type_metric.compute_f1(prefix='type-')
    trigger_result = trigger_metric.compute_f1(prefix='trigger-')
    event_result = event_metric.compute_f1(prefix='event-')
    role_result = role_metric.compute_f1(prefix='role-')
    argument_result = argument_metric.compute_f1(prefix='argument-')
    role_classify_result = role_classify_metric.compute_f1(prefix='role_classify-')

    result = dict()
    result.update(type_result)
    result.update(trigger_result)
    result.update(event_result)
    result.update(role_result)
    result.update(argument_result)
    result.update(role_classify_result)
    # result['AVG-F1'] = trigger_result.get('trigger-F1', 0.) + role_result.get('role-F1', 0.)
    result.update(counter)
    return result

def extract_target(text, regex):
    matches = re.findall(regex, text)
    if matches:
        target = matches
    else:
        target = ["None"]
    # print("target:", target)
    target = remove_none_values(target)
    return target


def extract_trigger_type(text, trigger_regex, type_regex):
    """ 从文本中提取 事件类型和事件触发词 """
    trigger_types = []
    # 提取所有的 <event> 标签及其内容
    events = split_string(text)
    for event in events:
        # print(f"event: {event}")
        event_type = extract_target(event, type_regex)
        # print(f"event_type: {event_type}")
        event_type = str(event_type).strip("[]' ")
        # print(f"event_type: {event_type}")
        trigger = extract_target(event, trigger_regex)
        # print(f"trigger: {trigger}")
        trigger = str(trigger).strip("[]' ")
        # print(f"trigger: {trigger}")
        # 组合事件类型、触发词
        if event_type and trigger:
            event = f"{event_type},{trigger}"
            trigger_types.append(event)
    trigger_types = remove_none_values(trigger_types)
    trigger_types = list(set(trigger_types))
    return trigger_types


def split_string(string, delimiter="<event>"):
    substrings = string.split(delimiter)
    cleaned_substrings = [delimiter + " " + substring.strip() for substring in substrings if substring.strip()]
    return cleaned_substrings


def extract_type_argument_role(text, type_regex, argument_regex, role_regex):
    """从文本中提取事件类型、论元和论元角色"""
    type_argument_roles = []
    # 提取所有的 <event> 标签及其内容
    events = split_string(text)
    for event in events:
        # print(f"event: {event}")
        event_type = extract_target(event, type_regex)
        arguments = extract_target(event, argument_regex)
        roles = extract_target(event, role_regex)
        event_type = str(event_type).strip("[]' ")
        for argument, role in zip(arguments, roles):
            # 组合事件类型、论元和论元角色
            if event_type and argument and role:
                temp = f"{event_type},{argument},{role}"
                type_argument_roles.append(temp)
    type_argument_roles = remove_none_values(type_argument_roles)
    type_argument_roles = list(set(type_argument_roles))
    return type_argument_roles


def remove_none_values(string_list):
    cleaned_list = [string for string in string_list if 'None' not in string]
    return cleaned_list


def eval_pred_linear(pred_lst, gold_lst):
    """
    Args:
        pred_lst:
        gold_lst: <event> breakdowns <trigger> Life:Divorce <type> Person <role> Welches <argument> <event> marriages
        <trigger> Life:Marry <type> Person <role> Welches <argument>
    Returns: Entity、relation和triple的precision、recall、f1-score
    """
    special_tokens = ['<event>', '<trigger>', '<type>', '<role>', '<argument>']
    event_type_metric = Metric()  # Event type Identification
    trigger_metric = Metric()  # Trigger Identification
    event_metric = Metric()  # Trigger Classification
    role_metric = Metric()  # Role Classification
    argument_metric = Metric()  # Argument Identification
    # argument_role_metric = Metric()  # Argument Role Identification
    type_argument_role_metric = Metric()  # Argument Classification

    for pred, gold in zip(pred_lst, gold_lst):
        if gold == "<event>":
            continue
        trigger_regex = r"\s+(\w+)\s+<trigger>"
        # 替换字符串 pred 开头的点号 .
        if pred.startswith("."):
            pred = "<event> " + pred[1:]
            print(f"pred: {pred}")
        pred_trigger = extract_target(pred, trigger_regex)
        gold_trigger = extract_target(gold, trigger_regex)
        trigger_metric.count_instance(gold_trigger, pred_trigger)
        type_regex = r"<trigger>\s*(.*?)\s*<type>"
        pred_type = extract_target(pred, type_regex)
        gold_type = extract_target(gold, type_regex)
        event_type_metric.count_instance(gold_type, pred_type)
        pred_event = extract_trigger_type(pred, trigger_regex, type_regex)
        gold_event = extract_trigger_type(gold, trigger_regex, type_regex)
        event_metric.count_instance(gold_event, pred_event)
        role_regex = r"\s+(\w+)\s+<role>"
        pred_role = extract_target(pred, role_regex)
        gold_role = extract_target(gold, role_regex)
        role_metric.count_instance(gold_role, pred_role)
        argument_regex = r"\s+(\w+)\s+<argument>"
        pred_argument = extract_target(pred, argument_regex)
        gold_argument = extract_target(gold, argument_regex)
        argument_metric.count_instance(gold_argument, pred_argument)
        pred_type_argument_role = extract_type_argument_role(pred, type_regex, argument_regex, role_regex)
        gold_type_argument_role = extract_type_argument_role(gold, type_regex, argument_regex, role_regex)
        type_argument_role_metric.count_instance(gold_type_argument_role, pred_type_argument_role)
    trigger_result = trigger_metric.compute_f1(prefix='trigger-')
    type_result = event_type_metric.compute_f1(prefix='event_type-')
    event_result = event_metric.compute_f1(prefix='event-')
    role_result = role_metric.compute_f1(prefix='role-')
    argument_result = argument_metric.compute_f1(prefix='argument-')
    type_argument_role_result = type_argument_role_metric.compute_f1(prefix='type_argument_role-')
    result = dict()
    result.update(trigger_result)
    result.update(type_result)
    result.update(event_result)
    result.update(role_result)
    result.update(argument_result)
    result.update(type_argument_role_result)
    print("result: ", result)
    return result


def get_extract_metrics(pred_lns: List[str], tgt_lns: List[str], decoding_format='tree'):
    if decoding_format == "tree":
        return eval_pred_tree(gold_list=tgt_lns, pred_list=pred_lns)
    else:
        return eval_pred_linear(pred_lst=pred_lns, gold_lst=tgt_lns)
