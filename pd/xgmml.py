import xml.etree.ElementTree as ET
import numpy as np

__xgmmltypes = {
    str: "string",
    int: "integer",
    float: "float",
    np.float64: "float",
    np.int64: "integer"
}
def savexgmml(sim, t, out):
    root = ET.Element("graph")
    tree = ET.ElementTree(root)

    root.set("xmlns", "http://www.cs.rpi.edu/XGMML")
    root.set("xmlns:dc", "http://purl.org/dc/elements/1.1/")
    root.set("xmlns:cy", "http://www.cytoscape.org")
    root.set("xmlns:rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#")
    root.set("xmlns:xlink", "http://www.w3.org/1999/xlink")
    root.set("directed", "0")

    def sattr(e, name, val):
        att = ET.Element("att")
        att.set("type", __xgmmltypes[type(val)])
        att.set("name", name)
        att.set("value", str(val))
        e.append(att)

    for n in sim.args.__dict__.keys():
        sattr(root, n, getattr(sim.args, n))

    e0, e1 = sim.entropy()
    sattr(root, "time", t)
    sattr(root, "e0", e0)
    sattr(root, "e1", e1)
    sattr(root, "avecoop", sim.avecoop)
    sattr(root, "avedegree", sim.avedegree)
    sattr(root, "aveprosp", sim.aveprosp)
    sattr(root, "tp", sim.tp)
    sattr(root, "tn", sim.tn)
    sattr(root, "fp", sim.fp)
    sattr(root, "fn", sim.fn)
    sattr(root, "ncasc", max(sim.ncascades.values()))
    sattr(root, "pcasc", max(sim.pcascades.values()))
    sattr(root, "transitions", sim.transitionNum)

    sattr(root, "backgroundColor", "#ffffff")

    for i in range(sim.N):
        node = ET.Element("node")
        node.set("label", str(i))
        node.set("id", str(i))
        sattr(node, "canonicalName", str(i))

        sattr(node, "node.size", "13.0")
        gr = ET.Element("graphics")
        if sim.kinds[i] == 0:
            sattr(node, "type", "cooperator")
            gr.set("fill", "#0000ff")
        else:
            sattr(node, "type", "defector")
            gr.set("fill", "#ff0000")

        gr.set("type", "ELLIPSE")
        gr.set("h", "13.0")
        gr.set("w", "13.0")
        gr.set("x", str(40.0 * i))
        gr.set("u", "0.0")
        gr.set("width", "1")
        gr.set("outline", "#666666")
        gr.set("cy:nodeTrransparency", "1.0")
        gr.set("cy:nodeLabelFont", "SansSerif.bold-0-12")
        gr.set("cy:borderLineType", "solid")

        node.append(gr)
        root.append(node)

        for j in sim.adj[i]:
            if j < i:
                continue
            edge = ET.Element("edge")
            name = "%s--%s" % (i, j)
            edge.set("label", name)
            edge.set("source", str(i))
            edge.set("target", str(j))
            sattr(edge, "canonicalName", name)
            sattr(edge, "interaction", "pp")
            gr = ET.Element("graphics")
            gr.set("width", "1")
            gr.set("fill", "#000000")
            gr.set("cy:sourceArrow", "0")
            gr.set("cy:targetArrow", "0")
            gr.set("cy:sourceArrowColor", "#000000")
            gr.set("cy:targetArrowColor", "#000000")
            gr.set("cy:edgeLabelFont", "Default-0-10")
            gr.set("cy:edgeLineType", "SOLID")
            gr.set("cy:curved", "STRAIGHT_LINES")
            edge.append(gr)
            root.append(edge)
    tree.write(out)

