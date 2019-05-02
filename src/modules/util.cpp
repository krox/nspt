#include "modules/util.h"

void MDeleteObject::run(Environment &env) { env.store.remove(params.name); }
