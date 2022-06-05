#include  <iostream>
#include "TFile.h"
#include "TTree.h"
#include "Pythia8/Pythia.h"

using namespace Pythia8;

//--------------------------------------------------------------------------

// Generic routine to extract the particles that existed right before
// the hadronization machinery was invoked.

void getPartonLevelEvent( Event& event, Event& partonLevelEvent) {

  // Copy over all particles that existed right before hadronization.
  partonLevelEvent.reset();
  for (int i = 0; i < event.size(); ++i)
  if (event[i].isFinalPartonLevel()) {
	// cout << "the status code of the parton level event is " << event[i].status() << endl;
	
    int iNew = partonLevelEvent.append( event[i] );

    // Set copied properties more appropriately: positive status,
    // original location as "mother", and with no daughters.
    partonLevelEvent[iNew].statusPos();
    partonLevelEvent[iNew].mothers( i, i);
    partonLevelEvent[iNew].daughters( 0, 0);
  }
}

//--------------------------------------------------------------------------

// Generic routine to extract the particles that exist after the
// hadronization machinery. Normally not needed, since SlowJet
// contains the standard possibilities preprogrammed, but this
// method illustrates further discrimination.

void getHadronLevelEvent( Event& event, Event& hadronLevelEvent) {

  // Iterate over all final particles.
  hadronLevelEvent.reset();
  for (int i = 0; i < event.size(); ++i) {
    bool accept = false;
    if (event[i].isFinal()) accept = true;

    // Copy over accepted particles, with original location as "mother".
    if (accept) {
      int iNew = hadronLevelEvent.append( event[i] );
// cout << "the status code of the hadron level event is " << event[i].status() << endl;

      hadronLevelEvent[iNew].mothers( i, i);
    }

  }

}
//--------------------------------------------------------------------------

int main(int argc, char* argv[]) {

TFile *Jetsoutput  = new TFile("Jetsoutput.root", "recreate");
TTree *tree = new TTree("tree", "tree");

int numJetsParton, numJetsHadron;
//numbers of parton jets in an event, number of hadron jets in an event
double pTPartonJets, pTHadronJets, yPartonJets, yHadronJets, phiPartonJets, phiHadronJets;

//Define tree branches
tree->Branch("numJetsParton", &numJetsParton, "numJetsParton/I");
tree->Branch("numJetsHadron", &numJetsHadron, "numJetsHadron/I");
tree->Branch("pTPartonJets", &pTPartonJets, "pTPartonJets/D");
tree->Branch("pTHadronJets", &pTHadronJets, "pTHadronJets/D");
tree->Branch("yPartonJets", &yPartonJets, "yPartonJets/D");
tree->Branch("yHadronJets", &yHadronJets, "yHadronJets/D");
tree->Branch("phiPartonJets", &phiPartonJets, "phiPartonJets/D");
tree->Branch("phiHadronJets", &phiHadronJets, "phiHadronJets/D");

  // Number of events, generated and listed ones.
  int nEvent    = 1000;
  int nListEvts = 1;
  int nListJets = 5;

  // Generator. LHC process and output selection. Initialization.
  Pythia pythia;

pythia.readFile(argv[1]);

//   pythia.readString("Beams:eCM = 13000.");
//   pythia.readString("HardQCD:all = on");
//   pythia.readString("PhaseSpace:pTHatMin = 200.");
//   pythia.readString("Next:numberShowInfo = 0");
//   pythia.readString("Next:numberShowProcess = 0");
//   pythia.readString("Next:numberShowEvent = 5");

  pythia.init();

  // Parton and Hadron Level event records. Remeber to initalize.
  Event partonLevelEvent;
  partonLevelEvent.init("Parton Level event record", &pythia.particleData);
  Event hadronLevelEvent;
  hadronLevelEvent.init("Hadron Level event record", &pythia.particleData);

  //  Parameters for the jet finders. Need select = 1 to catch partons.
  double radius   = 0.54;
  double pTjetMin = 10.;
  double etaMax   = 4.;
  int select      = 1;

  // Set up anti-kT clustering, comparing parton and hadron levels.
  SlowJet antiKTpartons( -1, radius, pTjetMin, etaMax, select);
  SlowJet antiKThadrons( -1, radius, pTjetMin, etaMax, select);

  // Begin event loop. Generate event. Skip if error.
  for (int iEvent = 0; iEvent < nEvent; ++iEvent) {
    if (!pythia.next()) continue;

    // Construct parton and hadron level event.
    getPartonLevelEvent( pythia.event, partonLevelEvent);
    getHadronLevelEvent( pythia.event, hadronLevelEvent);

    // List first few events.
    if (iEvent < nListEvts) {
      pythia.event.list();
      partonLevelEvent.list();
      hadronLevelEvent.list();
    }

    // Analyze jet properties and list first few analyses.
    antiKTpartons.analyze( partonLevelEvent );
    antiKThadrons.analyze( hadronLevelEvent );
    if (iEvent < nListJets) {
      antiKTpartons.list();
      antiKThadrons.list();
    }

// FILL TREE
numJetsParton = antiKTpartons.sizeJet() ;
numJetsHadron = antiKThadrons.sizeJet() ;
tree->Fill();

  for (int i = 0; i < antiKTpartons.sizeJet(); ++i) {
pTPartonJets = antiKTpartons.pT(i) ;
yPartonJets = antiKTpartons.y(i);
phiPartonJets = antiKTpartons.phi(i);
  tree->Fill();
  }

  for (int i = 0; i < antiKThadrons.sizeJet(); ++i) {
  pTHadronJets = antiKThadrons.pT(i) ;
  yHadronJets = antiKThadrons.y(i);
  phiHadronJets = antiKThadrons.phi(i);
  tree->Fill();
  }

  // End of event loop. Statistics. Histograms.
  }
  pythia.stat();

// Write the file and close it
Jetsoutput->Write();
Jetsoutput->Close();
  // Done.
  return 0;
}
