#include <faust/dsp/llvm-dsp.h>    // For LLVM backend
#include <faust/dsp/interpreter-dsp.h> // For Interpreter backend

using namespace std;

string faustCode = "import(\"stdfaust.lib\"); process = no.noise;";

// string errorString;
// llvm_dsp_factory* dspFactory = createDSPFactoryFromString(
//     "faust", faustCode, argc, argv, "", errorString, /*optimize=*/-1);

// if (!dspFactory) {
//     std::cerr << "Error creating DSP factory: " << errorString << std::endl;
//     return -1;
// }

// dsp* myDSP = dspFactory->createDSPInstance();
// if (!myDSP) {
//     std::cerr << "Error creating DSP instance." << std::endl;
//     deleteDSPFactory(dspFactory);
//     return -1;
// }

// myDSP->init(44100); 

// my_ui* ui = new my_ui();
// myDSP->buildUserInterface(ui);

// int numInputs = 2;
// int numOutputs = 2;
// int bufferSize = 128;

// float** input = new float*[numInputs];
// for (int i = 0; i < numInputs; ++i) {
//     input[i] = new float[bufferSize];
//     for (int j = 0; j < bufferSize; ++j) {
//         input[i][j] = 0.0f;
//     }
// }

// float** output = new float*[numOutputs];
// for (int i = 0; i < numOutputs; ++i) {
//     output[i] = new float[bufferSize];
// }

// while (running) { 
//     myDSP->compute(bufferSize, input, output);
// }


