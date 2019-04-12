#include "Grid/Grid.h"

#include "modules/module.h"
#include "nspt/grid_utils.h"
#include "util/CLI11.hpp"
#include "util/json.hpp"
#include "util/stopwatch.h"

using namespace nlohmann;
using namespace util;

int main(int argc, char **argv)
{
	Stopwatch swTotal;
	swTotal.start();

	Grid::Grid_init(&argc, &argv);

	std::vector<std::unique_ptr<Module>> modules;
	std::vector<Stopwatch> swModules;

	if (argc < 2 || argv[1][0] == '-')
	{
		if (primaryTask())
			fmt::print("Usage: {} <json-script> [Grid params...]\n", argv[0]);
		return -1;
	}

	// parse input json 'script'
	{
		std::string filename = argv[1];
		std::ifstream f(filename);
		json script;
		f >> script;

		for (const json &m : script.at("modules"))
		{
			std::string id;
			m.at("id").get_to(id);
			modules.push_back(createModule(id, m.at("params")));
		}
		if (primaryTask())
			fmt::print("found {} modules in script\n", modules.size());
		swModules.resize(modules.size());
	}

	// run modules
	Environment env;
	for (size_t i = 0; i < modules.size(); ++i)
	{
		if (primaryTask())
			fmt::print("==================== {} ====================\n",
			           modules[i]->name());
		swModules[i].start();
		modules[i]->run(env);
		swModules[i].stop();
		fmt::print("time for {}: {}\n", modules[i]->name(),
		           swModules[i].secs());
	}

	fmt::print("==================== Summary ====================\n");
	Grid::Grid_finalize();
	swTotal.stop();

	for (size_t i = 0; i < modules.size(); ++i)
		fmt::print("{}: {} secs ({:.2f} %)\n", modules[i]->name(),
		           swModules[i].secs(),
		           swModules[i].secs() / swTotal.secs() * 100.0);
	fmt::print("total: {} secs\n", swTotal.secs());
}
