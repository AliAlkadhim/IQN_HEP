#include "Pythia8/Pythia.h" // Include Pythia headers.
using namespace Pythia8;
int main(int argc, char* argv[]) {
// Set up generation.
Pythia pythia;

// Declare Pythia card file
 pythia.readFile(argv[1]);

pythia.init(); // Initialize; incoming pp beams is default.

// specify the number of events
int nevents = 5;

for (int i=0; i < nevents; i++) {

pythia.next(); // Generate an(other) event. Fill event record.

// calculate the number of particles in the event 
int nparticles = pythia.event.size();
cout << "Event: " << i << endl;
cout << "Event size: " << nparticles << endl;

 //START PARTICLE LOOP
 for (int j = 0; j < nparticles;  j++ ) {
     
    int id = pythia.event[j].id();
}
}
return 0;
}