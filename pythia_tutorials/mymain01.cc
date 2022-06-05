// Headers and Namespaces.
#include "Pythia8/Pythia.h" // Include Pythia headers.
using namespace Pythia8;
// Let Pythia8:: be implicit.
int main() {
// Begin main program.
// Set up generation.
Pythia pythia;
// Declare Pythia object
pythia.readString("Top:gg2ttbar = on"); // Switch on process.
pythia.readString("Beams:eCM = 8000."); // 8 TeV CM energy.
pythia.init(); // Initialize; incoming pp beams is default.
// Generate event(s).
pythia.next(); // Generate an(other) event. Fill event record.
return 0;
}
// End main program with error-free return.