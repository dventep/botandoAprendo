/* Edge Impulse ingestion SDK
 * Copyright (c) 2022 EdgeImpulse Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */
// If your target is limited in memory remove this macro to save 10K RAM
// #define EIDSP_QUANTIZE_FILTERBANK   0

/**
 * Define the number of slices per model window. E.g. a model window of 1000 ms
 * with slices per model window set to 4. Results in a slice size of 250 ms.
 * For more info: https://docs.edgeimpulse.com/docs/continuous-audio-sampling
 */
#define EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW 1

/*
 ** NOTE: If you run into TFLite arena allocation issue.
 **
 ** This may be due to may dynamic memory fragmentation.
 ** Try defining "-DEI_CLASSIFIER_ALLOCATION_STATIC" in boards.local.txt (create
 ** if it doesn't exist) and copy this file to
 ** `<ARDUINO_CORE_INSTALL_PATH>/arduino/hardware/<mbed_core>/<core_version>/`.
 **
 ** See
 ** (https://support.arduino.cc/hc/en-us/articles/360012076960-Where-are-the-installed-cores-located-)
 ** to find where Arduino installs cores on your machine.
 **
 ** If the problem persists then there's not enough memory for this model and application.
 */

/* Includes ---------------------------------------------------------------- */
#include <Servo.h>
#include <PDM.h>
#include <ia_embebidos_project_2_inferencing.h>

// Servo objects
Servo servoPlastico1;
Servo servoPapel2;
Servo servoDesecho3;
Servo servoOrganico4;
// Pin
const int servoPinPlastico1 = 2; // D2
const int servoPinPapel2 = 3; // D3
const int servoPinDesecho3 = 4; // D4
const int servoPinOrganico4 = 5; // D5

// Control variables (Angle)
int anglePlastico1 = 0;            // Plastico
int anglePapel2 = 0;            // Papel
int angleDesecho3 = 0;            // Desecho
int angleOrganico4 = 0;            // Organico
String prediction = "";
int timePassed = 0;

unsigned long previousMillis = 0;
const unsigned long interval = 2000;

/** Audio buffers, pointers and selectors */
typedef struct {
    signed short *buffers[2];
    unsigned char buf_select;
    unsigned char buf_ready;
    unsigned int buf_count;
    unsigned int n_samples;
} inference_t;

static inference_t inference;
static bool record_ready = false;
static signed short *sampleBuffer;
static bool debug_nn = false; // Set this to true to see e.g. features generated from the raw signal
static int print_results = -(EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW);

/**
 * @brief      Arduino setup function
 */
void setup()
{    
    servoPlastico1.attach(servoPinPlastico1);
    servoPapel2.attach(servoPinPapel2);
    servoDesecho3.attach(servoPinDesecho3);
    servoOrganico4.attach(servoPinOrganico4);

    servoPlastico1.write(anglePlastico1);
    servoPapel2.write(anglePapel2);
    servoDesecho3.write(angleDesecho3);
    servoOrganico4.write(angleOrganico4);
    // put your setup code here, to run once:
    Serial.begin(115200);
    // comment out the below line to cancel the wait for USB connection (needed for native USB)
    while (!Serial);
    Serial.println("Edge Impulse Inferencing Demo");

    // summary of inferencing settings (from model_metadata.h)
    ei_printf("Inferencing settings:\n");
    ei_printf("\tInterval: %.2f ms.\n", (float)EI_CLASSIFIER_INTERVAL_MS);
    ei_printf("\tFrame size: %d\n", EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE);
    ei_printf("\tSample length: %d ms.\n", EI_CLASSIFIER_RAW_SAMPLE_COUNT / 16);
    ei_printf("\tNo. of classes: %d\n", sizeof(ei_classifier_inferencing_categories) /
                                            sizeof(ei_classifier_inferencing_categories[0]));

    run_classifier_init();
    if (microphone_inference_start(EI_CLASSIFIER_SLICE_SIZE) == false) {
        ei_printf("ERR: Could not allocate audio buffer (size %d), this could be due to the window length of your model\r\n", EI_CLASSIFIER_RAW_SAMPLE_COUNT);
        return;
    }
}

/**
 * @brief      Arduino main function. Runs the inferencing loop.
 */
void loop()
{
    if (prediction != "") {
        timePassed = timePassed + 1;
        digitalWrite(LED_BUILTIN, LOW);
    }
    if (timePassed > 4) {
        timePassed = 0;
        prediction = "";
    }
    if (prediction == "") {        
        digitalWrite(LED_BUILTIN, HIGH);
    }
    bool m = microphone_inference_record();
    if (!m) {
        ei_printf("ERR: Failed to record audio...\n");
        return;
    }

    signal_t signal;
    signal.total_length = EI_CLASSIFIER_SLICE_SIZE;
    signal.get_data = &microphone_audio_signal_get_data;
    ei_impulse_result_t result = {0};

    EI_IMPULSE_ERROR r = run_classifier_continuous(&signal, &result, debug_nn);
    int predictedIndex = 0;
    const char* labelDetected = "Desconocido";
    float maxVal = result.classification[0].value;

    if (r != EI_IMPULSE_OK) {
        ei_printf("ERR: Failed to run classifier (%d)\n", r);
        return;
    }

    if (++print_results >= (EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW) && timePassed == 0) {
        for (size_t ix = 0; ix < EI_CLASSIFIER_LABEL_COUNT; ix++) {
            ei_printf("    %s: %.5f\n", result.classification[ix].label,
                    result.classification[ix].value);
            if (result.classification[ix].value > maxVal) {
                maxVal = result.classification[ix].value;
                labelDetected = result.classification[ix].label;
                predictedIndex = ix;
            }
        }
        #if EI_CLASSIFIER_HAS_ANOMALY == 1
            ei_printf("    anomaly score: %.3f - timePassed: %d\n", result.anomaly, timePassed);
        #endif

        if ((result.anomaly > -2.5 && result.anomaly < 0.0) || result.anomaly > 0.7) {
            ei_printf("Anomalía detectada. No ejecutar ninguna acción - timePassed: %d.\n", timePassed);
        } else {
            
            if (maxVal >= 0.4 && predictedIndex >= 0) {
                String pred = String(labelDetected);
                prediction = pred;

                if (pred == "Plastico") {
                    ei_printf("♻️ Acción: Clasificar como Plastico\n");
                    servoPlastico1.write(90);
                    servoPapel2.write(0);
                    servoDesecho3.write(0);
                    servoOrganico4.write(0);
                }
                else if (pred == "Papel") {
                    ei_printf("🗑️ Acción: Clasificar como Papel\n");
                    servoPlastico1.write(0);
                    servoPapel2.write(90);
                    servoDesecho3.write(0);
                    servoOrganico4.write(0);
                }
                else if (pred == "Desecho") {
                    ei_printf("📄 Acción: Clasificar como Desecho\n");
                    servoPlastico1.write(0);
                    servoPapel2.write(0);
                    servoDesecho3.write(90);
                    servoOrganico4.write(0);
                }
                else if (pred == "Organico") {
                    ei_printf("🧴 Acción: Clasificar como Organico\n");
                    servoPlastico1.write(0);
                    servoPapel2.write(0);
                    servoDesecho3.write(0);
                    servoOrganico4.write(90);
                }
                else if (pred == "Pilas") {
                    ei_printf("🔋 Acción: Clasificar como Pilas\n");
                    servoPlastico1.write(0);
                    servoPapel2.write(0);
                    servoDesecho3.write(0);
                    servoOrganico4.write(0);
                }
            } else {
                ei_printf("Ninguna clase supera el umbral. No ejecutar acción.\n");
            }
        }

        print_results = 0;
    }
}

/**
 * @brief      PDM buffer full callback
 *             Get data and call audio thread callback
 */
static void pdm_data_ready_inference_callback(void)
{
    int bytesAvailable = PDM.available();

    // read into the sample buffer
    int bytesRead = PDM.read((char *)&sampleBuffer[0], bytesAvailable);

    if (record_ready == true) {
        for (int i = 0; i<bytesRead>> 1; i++) {
            inference.buffers[inference.buf_select][inference.buf_count++] = sampleBuffer[i];

            if (inference.buf_count >= inference.n_samples) {
                inference.buf_select ^= 1;
                inference.buf_count = 0;
                inference.buf_ready = 1;
            }
        }
    }
}

/**
 * @brief      Init inferencing struct and setup/start PDM
 *
 * @param[in]  n_samples  The n samples
 *
 * @return     { description_of_the_return_value }
 */
static bool microphone_inference_start(uint32_t n_samples)
{
    inference.buffers[0] = (signed short *)malloc(n_samples * sizeof(signed short));

    if (inference.buffers[0] == NULL) {
        return false;
    }

    inference.buffers[1] = (signed short *)malloc(n_samples * sizeof(signed short));

    if (inference.buffers[1] == NULL) {
        free(inference.buffers[0]);
        return false;
    }

    sampleBuffer = (signed short *)malloc((n_samples >> 1) * sizeof(signed short));

    if (sampleBuffer == NULL) {
        free(inference.buffers[0]);
        free(inference.buffers[1]);
        return false;
    }

    inference.buf_select = 0;
    inference.buf_count = 0;
    inference.n_samples = n_samples;
    inference.buf_ready = 0;

    // configure the data receive callback
    PDM.onReceive(&pdm_data_ready_inference_callback);

    PDM.setBufferSize((n_samples >> 1) * sizeof(int16_t));

    // initialize PDM with:
    // - one channel (mono mode)
    // - a 16 kHz sample rate
    if (!PDM.begin(1, EI_CLASSIFIER_FREQUENCY)) {
        ei_printf("Failed to start PDM!");
    }

    // set the gain, defaults to 20
    PDM.setGain(127);

    record_ready = true;

    return true;
}

/**
 * @brief      Wait on new data
 *
 * @return     True when finished
 */
static bool microphone_inference_record(void)
{
    bool ret = true;

    if (inference.buf_ready == 1) {
        ei_printf(
            "Error sample buffer overrun. Decrease the number of slices per model window "
            "(EI_CLASSIFIER_SLICES_PER_MODEL_WINDOW)\n");
        ret = false;
    }

    while (inference.buf_ready == 0) {
        delay(1);
    }

    inference.buf_ready = 0;

    return ret;
}

/**
 * Get raw audio signal data
 */
static int microphone_audio_signal_get_data(size_t offset, size_t length, float *out_ptr)
{
    numpy::int16_to_float(&inference.buffers[inference.buf_select ^ 1][offset], out_ptr, length);

    return 0;
}

/**
 * @brief      Stop PDM and release buffers
 */
static void microphone_inference_end(void)
{
    PDM.end();
    free(inference.buffers[0]);
    free(inference.buffers[1]);
    free(sampleBuffer);
}

#if !defined(EI_CLASSIFIER_SENSOR) || EI_CLASSIFIER_SENSOR != EI_CLASSIFIER_SENSOR_MICROPHONE
#error "Invalid model for current sensor."
#endif
