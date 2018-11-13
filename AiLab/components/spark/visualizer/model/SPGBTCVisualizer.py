# coding=utf-8
from __future__ import print_function

import json
import xml.etree.ElementTree as ET
from io import StringIO

from jinja2 import Template
from suanpan.arguments import String
from suanpan.spark import SparkComponent as sc
from suanpan.spark.arguments import PmmlModel, Visual


def load_pmml(pmml):
    root = load_as_xml(pmml, namespace=False)
    return parse(root)


def load_as_xml(pmml, namespace=True):
    it = ET.iterparse(StringIO(pmml))
    if not namespace:
        remove_namespaces(it)
    return it.root


def remove_namespaces(it):
    for _, el in it:
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]  # strip all namespaces
    return it


def parse(element):
    return [parse_tree(tree) for tree in element.findall(".//TreeModel/..")]


def parse_tree(segment):
    tree_dom = segment.find("./TreeModel")
    node_dom = tree_dom.find("./Node")
    if node_dom is None:
        raise Exception("No node in TreeModel")

    tree = parse_node(node_dom)
    return dict(weight=segment.get("weight"), tree=true_false_to_children(tree))


def true_false_to_children(tree):
    true_node, false_node = tree.pop("True", None), tree.pop("False", None)
    if true_node:
        true_node = true_false_to_children(true_node)
    if false_node:
        false_node = true_false_to_children(false_node)
    tree["children"] = [node for node in [true_node, false_node] if node]
    return tree


def parse_node(node_dom):
    node = parse_result_node(node_dom)

    children_node_doms = node_dom.findall("./Node")
    if children_node_doms:
        first_child_dom = children_node_doms[0]
        predicate_dom = first_child_dom.find("./SimplePredicate")
        tree = parse_predicate_node(predicate_dom)
        current_tree = tree
        for dom in children_node_doms[1:]:
            predicate_dom = dom.find("./SimplePredicate")
            predicate_node = parse_predicate_node(predicate_dom)
            current_tree["False"] = predicate_node
            current_tree = predicate_node
        current_tree["False"] = node
        current_tree = tree
        for dom in children_node_doms:
            child_tree = parse_node(dom)
            current_tree["True"] = child_tree
            current_tree = current_tree["False"]
        return tree
    else:
        return node


def parse_predicate_node(precition):
    string = " ".join(
        [
            precition.get("field"),
            operator(precition.get("operator")),
            str(round(float(precition.get("value")), 2)),
        ]
    )
    return dict(type="predicateNode", name=string, **dict(precition.items()))


def parse_result_node(node):
    score = node.get("score")
    return dict(type="resultNode", value=score, name=round(float(score), 2))


def operator(string):
    return {"lessOrEqual": "<=", "less": "<", "moreOrEqual": ">=", "more": ">"}[string]


template = Template(
    """
///define:type=echarts3
///define:title={{ title }}

data = {{ data }};

colors = ['#2196f3', '#ae5da1', '#f0a332', '#73dabe', '#8fc31f']
option = {
    tooltip: {
        trigger: "item",
        triggerOn: "mousemove",
        padding: [6, 12],
        backgroundColor: 'rgba(0,0,0,0.7)',
        textStyle: {
            color: 'rgba(255, 255, 255, 0.8)',
            fontSize: 12,
        },
        formatter: function(params) {
            var nodeData = params.data;
            switch(nodeData.type) {
                case "predicateNode":
                    return "True | False"
                case "resultNode":
                    return "Score: " + nodeData.value;
            }
        }
    },
    legend: {
        type: 'scroll',
        orient: 'vertical',
        right: 0,
        top: 'center',
        data: data.map(function(tree, index, array) {
            var name = "Tree" + (index + 1) + ': ' + tree.weight;
            return {
                name: name,
                icon: "roundRect",
            };
        }),
        selectedMode: "single",
        itemWidth: 24,
        itemHeight: 12,
        itemGap: 16,
        padding: [20, 12],
        textStyle: {
            color: 'rgba(0,0,0,0.65)'
        },
        pageIconColor: '#888',
        pageIconInactiveColor: '#bbb',
        pageIconSize: 12
    },
    series: data.map(function(tree, index, array) {
        var color = colors[index % colors.length];
        var _tree = tree.tree;
        return {
            type: "tree",
            roam: true,
            orient: "vertical",
            name: "Tree" + (index + 1) + ': ' + tree.weight,
            data: [_tree],
            top: 20,
            bottom: 40,
            left: 20,
            right: 120,
            initialTreeDepth: 6,
            symbolSize: 12,
            symbol: 'circle',
            itemStyle: {
                normal: {
                    color: color,
                    borderColor: '#fff',
                    shadowColor: 'rgb(64, 64, 64, 0.5)',
                    shadowBlur: 4,
                },
            },
            lineStyle: {
                color: '#d9d9d9',
                width: 1
            },
            label: {
                normal: {
                    color: 'rgba(0, 0, 0, 0.65)',
                    position: "right",
                    distance: 8,
                    verticalAlign: "middle",
                    backgroundColor: '#f0f0f0',
                    borderWidth: 1,
                    borderColor: '#d9d9d9',
                    borderRadius: 3,
                    borderWidth: 1,
                    padding: [4, 8]
                }
            },
            expandAndCollapse: true,
            animationDuration: 550,
            animationDurationUpdate: 750
        };
    })
};
"""
)


@sc.input(PmmlModel(key="inputModel"))
@sc.output(Visual(key="outputVisual"))
@sc.param(String(key="title", default="GBTC"))
def SPGBTCVisualizer(context):
    args = context.args

    data = json.dumps(load_pmml(args.inputModel))
    return template.render(title=args.title, data=data)


if __name__ == "__main__":
    SPGBTCVisualizer()  # pylint: disable=no-value-for-parameter
