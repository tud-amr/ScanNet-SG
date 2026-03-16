// topology_map.h
#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <memory>
#include <stdexcept>
#include <json/single_include/nlohmann/json.hpp>
#include <Eigen/Dense>
#include <iostream>
#include <fstream>

using json = nlohmann::json;
using namespace Eigen;

enum class ShapeType { CYLINDER, ORIENTED_BOX };

struct Orientation {
    float x, y, z, w;

    Eigen::Quaternionf toEigenQuaternion() const {
        return Eigen::Quaternionf(w, x, y, z);
    }

    Eigen::Matrix3f toRotationMatrix() const {
        return toEigenQuaternion().toRotationMatrix();
    }

};

struct Shape {
    virtual ~Shape() = default;
    virtual ShapeType type() const = 0;
    virtual json to_json() const = 0;
    static std::shared_ptr<Shape> from_json(const json& j);
};

struct Cylinder : public Shape {
    float radius;
    float height;
    Orientation orientation;

    Cylinder(float r, float h, Orientation o) : radius(r), height(h), orientation(o) {}

    ShapeType type() const override { return ShapeType::CYLINDER; }

    json to_json() const override {
        return {
            {"radius", radius},
            {"height", height},
            {"orientation", {{"x", orientation.x}, {"y", orientation.y}, {"z", orientation.z}, {"w", orientation.w}}}
        };
    }

    static Cylinder from_json(const json& j) {
        Orientation o{
            j["orientation"]["x"], j["orientation"]["y"],
            j["orientation"]["z"], j["orientation"]["w"]
        };
        return Cylinder(j.at("radius"), j.at("height"), o);
    }
};



struct OrientedBox : public Shape {
    float length, width, height;
    Orientation orientation;

    OrientedBox(float l, float w, float h, Orientation o)
        : length(l), width(w), height(h), orientation(o) {}

    ShapeType type() const override { return ShapeType::ORIENTED_BOX; }

    json to_json() const override {
        return {
            {"length", length},
            {"width", width},
            {"height", height},
            {"orientation", {{"x", orientation.x}, {"y", orientation.y}, {"z", orientation.z}, {"w", orientation.w}}}
        };
    }

    static OrientedBox from_json(const json& j) {
        Orientation o{
            j["orientation"]["x"], j["orientation"]["y"],
            j["orientation"]["z"], j["orientation"]["w"]
        };
        return OrientedBox(j.at("length"), j.at("width"), j.at("height"), o);
    }
};


inline std::shared_ptr<Shape> Shape::from_json(const json& j) {
    if (j.contains("radius")) {
        return std::make_shared<Cylinder>(Cylinder::from_json(j));
    } else {
        return std::make_shared<OrientedBox>(OrientedBox::from_json(j));
    }
}

// Node base class
struct Node {
    std::string id;
    Vector3f position{0, 0, 0}; // test only

    virtual json to_json() const = 0;
    virtual ~Node() = default;
};

// ObjectNode
struct ObjectNode : public Node {
    std::string name;
    VectorXf visual_embedding;
    VectorXf text_embedding;
    std::shared_ptr<Shape> shape;

    json to_json() const override {
        return {
            {"id", id},
            {"name", name},
            {"visual_embedding", std::vector<float>(visual_embedding.data(), visual_embedding.data() + visual_embedding.size())},
            {"text_embedding", std::vector<float>(text_embedding.data(), text_embedding.data() + text_embedding.size())},
            {"shape", shape->to_json()},
            {"position", {position[0], position[1], position[2]}}
        };
    }

    static ObjectNode from_json(const json& j) {
        ObjectNode node;
        node.id = j.at("id");
        node.name = j.at("name");

        std::vector<float> visual = j.at("visual_embedding");
        node.visual_embedding = Eigen::Map<VectorXf>(visual.data(), visual.size());

        std::vector<float> text = j.at("text_embedding");
        node.text_embedding = Eigen::Map<VectorXf>(text.data(), text.size());

        node.shape = Shape::from_json(j.at("shape"));

        auto pos = j.at("position");
        node.position = Vector3f(pos[0], pos[1], pos[2]);
        return node;
    }
};

// FreeSpaceNode
struct FreeSpaceNode : public Node {
    float radius;

    json to_json() const override {
        return {
            {"id", id},
            {"radius", radius},
            {"position", {position[0], position[1], position[2]}}
        };
    }

    static FreeSpaceNode from_json(const json& j) {
        FreeSpaceNode node;
        node.id = j.at("id");
        node.radius = j.at("radius");
        auto pos = j.at("position");
        node.position = Vector3f(pos[0], pos[1], pos[2]);
        return node;
    }
};

// Edge
struct Edge {
    std::string source_id, target_id;
    float distance;
    Vector3f direction;
    std::string description;

    json to_json() const {
        return {
            {"source_id", source_id},
            {"target_id", target_id},
            {"distance", distance},
            {"direction", {direction[0], direction[1], direction[2]}},
            {"description", description}
        };
    }

    static Edge from_json(const json& j) {
        Edge e;
        e.source_id = j.at("source_id");
        e.target_id = j.at("target_id");
        e.distance = j.at("distance");
        auto dir = j.at("direction");
        e.direction = Vector3f(dir[0], dir[1], dir[2]);
        e.description = j.at("description");
        return e;
    }
};

// Edge Hypothesis
struct TopologyMapHypothesis {
    std::string id;
    float confidence;
    std::unordered_map<std::string, Edge> edges;

    json to_json() const {
        json edge_json;
        for (const auto& [eid, edge] : edges) {
            edge_json[eid] = edge.to_json();
        }
        return {{"id", id}, {"confidence", confidence}, {"edges", edge_json}};
    }

    static TopologyMapHypothesis from_json(const json& j) {
        TopologyMapHypothesis h;
        h.id = j.at("id");
        h.confidence = j.at("confidence");
        for (auto& [eid, edge_data] : j.at("edges").items()) {
            h.edges[eid] = Edge::from_json(edge_data);
        }
        return h;
    }
};

// TopologyMap
struct TopologyMap {
    std::unordered_map<std::string, ObjectNode> object_nodes;
    std::unordered_map<std::string, FreeSpaceNode> free_space_nodes;
    std::unordered_map<std::string, TopologyMapHypothesis> edge_hypotheses;

    json to_json() const {
        json obj, free, hypo;
        for (const auto& [id, node] : object_nodes) obj["nodes"][id] = node.to_json();
        for (const auto& [id, node] : free_space_nodes) free["nodes"][id] = node.to_json();
        for (const auto& [id, hyp] : edge_hypotheses) hypo[id] = hyp.to_json();

        return {{"object_nodes", obj}, {"free_space_nodes", free}, {"edge_hypotheses", hypo}};
    }

    void from_json(const json& j) {
        object_nodes.clear();
        free_space_nodes.clear();
        edge_hypotheses.clear();

        // Check if object_nodes exists and has a "nodes" subfield
        if (j.contains("object_nodes") && j["object_nodes"].contains("nodes")) {
            for (const auto& [id, node_data] : j["object_nodes"]["nodes"].items()) {
                object_nodes[id] = ObjectNode::from_json(node_data);
            }
            std::cout << "Loaded " << object_nodes.size() << " object nodes" << std::endl;
        } else {
            std::cout << "No object nodes found in the json file" << std::endl;
        }

        // Check if free_space_nodes exists and has a "nodes" subfield
        if (j.contains("free_space_nodes") && j["free_space_nodes"].contains("nodes")) {
            for (const auto& [id, node_data] : j["free_space_nodes"]["nodes"].items()) {
                free_space_nodes[id] = FreeSpaceNode::from_json(node_data);
            }
            std::cout << "Loaded " << free_space_nodes.size() << " free space nodes" << std::endl;
        } else {
            std::cout << "No free space nodes found in the json file" << std::endl;
        }

        // Check if edge_hypotheses exists
        if (j.contains("edge_hypotheses")) {
            for (const auto& [id, hypo_data] : j["edge_hypotheses"].items()) {
                edge_hypotheses[id] = TopologyMapHypothesis::from_json(hypo_data);
            }
            std::cout << "Loaded " << edge_hypotheses.size() << " edge hypotheses" << std::endl;
        } else {
            std::cout << "No edge hypotheses found in the json file" << std::endl;
        }
    }

    void save(const std::string& path) const {
        std::ofstream file(path);
        file << to_json().dump(4);
    }

    void load(const std::string& path) {
        std::ifstream file(path);
        from_json(json::parse(file));
    }

    std::string serialize() const {
        return to_json().dump(4);
    }

    void deserialize(const std::string& s) {
        from_json(json::parse(s));
    }
};
