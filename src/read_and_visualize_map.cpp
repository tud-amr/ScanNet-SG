#include "topology_map.h"
#include "utils.h"
#include <string>


int main(int argc, char** argv) {

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <map_file>" << std::endl;
        return 1;
    }

    std::string map_file = argv[1];
    TopologyMap map;
    map.load(map_file);
    showTopologyMap(map);
}
