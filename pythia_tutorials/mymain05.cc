// Headers and Namespaces.
#include "Pythia8/Pythia.h" // Include Pythia headers.
using namespace Pythia8;
// Let Pythia8:: be implicit.
int main(int argc, char* argv[]) {
// Begin main program.
// Set up generation.
Pythia pythia;
// Declare Pythia object
pythia.readFile(argv[1]);

pythia.init(); // Initialize; incoming pp beams is default.
// Generate event(s).
pythia.next(); // Generate an(other) event. Fill event record.
return 0;
}
