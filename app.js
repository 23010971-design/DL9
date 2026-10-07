// ============================================================
// YOLO11 SEGMENTATION REALTIME WEB APP
// iPhone Camera + ONNX Runtime Web
// MASK ONLY - NO BOUNDING BOX
// ============================================================

const MODEL_PATH = "./best.onnx";

const INPUT_SIZE = 320;

const CONF_THRESHOLD = 0.60;
const IOU_THRESHOLD = 0.50;
const MASK_THRESHOLD = 0.50;

const INFERENCE_INTERVAL = 30;


// ============================================================
// CLASS NAMES
// ============================================================

const CLASS_NAMES = [
    "light",              // ID 0
    "switch",             // ID 1
    "emergency_light",    // ID 2
    "camera",             // ID 3
    "volume"              // ID 4
];


// ============================================================
// CLASS COLORS
// ============================================================

const CLASS_COLORS = [
    [255, 255, 0],     // 0 = light
    [0, 200, 255],     // 1 = switch
    [255, 165, 0],     // 2 = emergency_light
    [0, 255, 0],       // 3 = camera
    [200, 80, 255]     // 4 = volume
];


// ============================================================
// GLOBAL VARIABLES
// ============================================================

let session = null;
let stream = null;

let running = false;
let inferenceBusy = false;

let lastInferenceTime = 0;

let frameCounter = 0;
let lastFPSUpdate = performance.now();

let outputsLogged = false;


// ============================================================
// DOM
// ============================================================

const video =
    document.getElementById("video");

const overlay =
    document.getElementById("overlay");

const overlayCtx =
    overlay.getContext("2d");

const processingCanvas =
    document.getElementById("processingCanvas");

const processingCtx =
    processingCanvas.getContext(
        "2d",
        {
            willReadFrequently: true
        }
    );

const statusElement =
    document.getElementById("status");

const fpsElement =
    document.getElementById("fps");

const objectsElement =
    document.getElementById("objects");

const detectionsElement =
    document.getElementById("detections");

const loadButton =
    document.getElementById("loadButton");

const cameraButton =
    document.getElementById("cameraButton");

const stopButton =
    document.getElementById("stopButton");


// ============================================================
// STATUS
// ============================================================

function setStatus(text) {

    if (statusElement) {
        statusElement.textContent = text;
    }

    console.log(
        "[SEGMENTATION]",
        text
    );
}


// ============================================================
// LOAD MODEL
// ============================================================

async function loadModel() {

    if (session) {

        setStatus(
            "Segmentation model already loaded"
        );

        return true;
    }

    try {

        setStatus(
            "Loading segmentation model..."
        );

        if (loadButton) {
            loadButton.disabled = true;
        }


        // ----------------------------------------------------
        // TRY WEBGL
        // ----------------------------------------------------

        try {

            session =
                await ort.InferenceSession.create(
                    MODEL_PATH,
                    {
                        executionProviders: [
                            "webgl"
                        ],
                        graphOptimizationLevel:
                            "all"
                    }
                );

            setStatus(
                "Segmentation loaded - WebGL"
            );

        }

        catch (webglError) {

            console.warn(
                "WebGL failed:",
                webglError
            );


            // ------------------------------------------------
            // WASM FALLBACK
            // ------------------------------------------------

            session =
                await ort.InferenceSession.create(
                    MODEL_PATH,
                    {
                        executionProviders: [
                            "wasm"
                        ],
                        graphOptimizationLevel:
                            "all"
                    }
                );

            setStatus(
                "Segmentation loaded - WASM"
            );
        }


        console.log(
            "INPUT NAMES:",
            session.inputNames
        );

        console.log(
            "OUTPUT NAMES:",
            session.outputNames
        );

        console.log(
            "INPUT METADATA:",
            session.inputMetadata
        );

        console.log(
            "OUTPUT METADATA:",
            session.outputMetadata
        );


        return true;

    }

    catch (error) {

        console.error(
            "MODEL ERROR:",
            error
        );

        session = null;

        if (loadButton) {
            loadButton.disabled = false;
        }

        setStatus(
            "Model loading failed"
        );

        alert(
            "Không load được best.onnx.\n\n" +
            "Hãy kiểm tra best.onnx có phải " +
            "YOLO Segmentation model hay không."
        );

        return false;
    }
}


// ============================================================
// START CAMERA
// ============================================================

async function startCamera() {

    if (!session) {

        const loaded =
            await loadModel();

        if (!loaded) {
            return;
        }
    }


    try {

        // ----------------------------------------------------
        // STOP OLD STREAM
        // ----------------------------------------------------

        if (stream) {

            stream
                .getTracks()
                .forEach(
                    track =>
                        track.stop()
                );

            stream = null;
        }


        setStatus(
            "Requesting rear camera..."
        );


        // ----------------------------------------------------
        // IPHONE REAR CAMERA
        // ----------------------------------------------------

        stream =
            await navigator
                .mediaDevices
                .getUserMedia({

                    video: {

                        facingMode: {
                            ideal: "environment"
                        },

                        width: {
                            ideal: 640
                        },

                        height: {
                            ideal: 480
                        },

                        frameRate: {
                            ideal: 30,
                            max: 30
                        }
                    },

                    audio: false
                });


        video.srcObject =
            stream;

        await video.play();

        setupOverlay();

        running = true;
        inferenceBusy = false;
        lastInferenceTime = 0;

        setStatus(
            "Segmentation camera running"
        );

        requestAnimationFrame(
            inferenceLoop
        );

    }

    catch (error) {

        console.error(
            "CAMERA ERROR:",
            error
        );

        setStatus(
            "Camera error"
        );

        alert(
            "Không mở được camera.\n\n" +
            "Hãy dùng HTTPS và cho phép Safari " +
            "truy cập camera."
        );
    }
}


// ============================================================
// STOP CAMERA
// ============================================================

function stopCamera() {

    running = false;
    inferenceBusy = false;

    if (stream) {

        stream
            .getTracks()
            .forEach(
                track =>
                    track.stop()
            );

        stream = null;
    }

    video.srcObject = null;

    overlayCtx.clearRect(
        0,
        0,
        overlay.width,
        overlay.height
    );

    objectsElement.textContent =
        "0";

    fpsElement.textContent =
        "0";

    detectionsElement.textContent =
        "No detection";

    setStatus(
        session
            ? "Model loaded"
            : "Model not loaded"
    );
}


// ============================================================
// OVERLAY
// ============================================================

function setupOverlay() {

    if (
        !video.videoWidth ||
        !video.videoHeight
    ) {
        return;
    }

    overlay.width =
        video.videoWidth;

    overlay.height =
        video.videoHeight;
}


// ============================================================
// PREPROCESS IMAGE
// ============================================================

function preprocess() {

    const videoWidth =
        video.videoWidth;

    const videoHeight =
        video.videoHeight;


    if (
        !videoWidth ||
        !videoHeight
    ) {
        return null;
    }


    processingCanvas.width =
        INPUT_SIZE;

    processingCanvas.height =
        INPUT_SIZE;


    processingCtx.clearRect(
        0,
        0,
        INPUT_SIZE,
        INPUT_SIZE
    );


    // --------------------------------------------------------
    // LETTERBOX
    // --------------------------------------------------------

    const scale =
        Math.min(
            INPUT_SIZE / videoWidth,
            INPUT_SIZE / videoHeight
        );


    const newWidth =
        videoWidth * scale;

    const newHeight =
        videoHeight * scale;


    const offsetX =
        (INPUT_SIZE - newWidth) / 2;

    const offsetY =
        (INPUT_SIZE - newHeight) / 2;


    processingCtx.fillStyle =
        "#808080";

    processingCtx.fillRect(
        0,
        0,
        INPUT_SIZE,
        INPUT_SIZE
    );


    processingCtx.drawImage(
        video,

        0,
        0,
        videoWidth,
        videoHeight,

        offsetX,
        offsetY,
        newWidth,
        newHeight
    );


    const imageData =
        processingCtx.getImageData(
            0,
            0,
            INPUT_SIZE,
            INPUT_SIZE
        );


    const pixels =
        imageData.data;


    const pixelCount =
        INPUT_SIZE *
        INPUT_SIZE;


    const floatData =
        new Float32Array(
            pixelCount * 3
        );


    // --------------------------------------------------------
    // RGB -> CHW
    // --------------------------------------------------------

    for (
        let i = 0;
        i < pixelCount;
        i++
    ) {

        const p =
            i * 4;


        floatData[i] =
            pixels[p] / 255.0;


        floatData[
            pixelCount + i
        ] =
            pixels[p + 1] / 255.0;


        floatData[
            pixelCount * 2 + i
        ] =
            pixels[p + 2] / 255.0;
    }


    return {

        tensor:
            new ort.Tensor(
                "float32",
                floatData,
                [
                    1,
                    3,
                    INPUT_SIZE,
                    INPUT_SIZE
                ]
            ),

        scale,

        offsetX,

        offsetY
    };
}


// ============================================================
// RUN INFERENCE
// ============================================================

async function runInference() {

    const prep =
        preprocess();


    if (!prep) {
        return null;
    }


    const inputName =
        session.inputNames[0];


    const feeds = {};

    feeds[inputName] =
        prep.tensor;


    const results =
        await session.run(
            feeds
        );


    if (!outputsLogged) {

        outputsLogged = true;

        console.log(
            "================================"
        );

        console.log(
            "YOLO SEGMENTATION OUTPUTS"
        );

        console.log(
            "================================"
        );


        for (
            const name
            of Object.keys(results)
        ) {

            console.log(
                name,
                results[name].dims
            );
        }
    }


    const outputs =
        session.outputNames.map(
            name =>
                results[name]
        );


    const detectionOutput =
        findDetectionOutput(
            outputs
        );


    const prototypeOutput =
        findPrototypeOutput(
            outputs
        );


    if (!detectionOutput) {

        throw new Error(
            "Không tìm thấy detection output."
        );
    }


    if (!prototypeOutput) {

        throw new Error(
            "Không tìm thấy MASK prototype output. " +
            "Model có thể không phải segmentation."
        );
    }


    return {

        detectionOutput,

        prototypeOutput,

        scale:
            prep.scale,

        offsetX:
            prep.offsetX,

        offsetY:
            prep.offsetY
    };
}


// ============================================================
// FIND DETECTION OUTPUT
// ============================================================

function findDetectionOutput(outputs) {

    return outputs.find(output => {

        if (!output || !output.dims) {
            return false;
        }

        const d = output.dims;

        if (d.length !== 3 || d[0] !== 1) {
            return false;
        }

        const dim1 = d[1];
        const dim2 = d[2];

        // [1, channels, boxes]
        const channelFirst =
            dim1 > 4 &&
            dim1 <= 256 &&
            dim2 > 100;

        // [1, boxes, channels]
        const boxFirst =
            dim1 > 100 &&
            dim2 > 4 &&
            dim2 <= 256;

        return channelFirst || boxFirst;

    }) || null;
}


// ============================================================
// FIND MASK PROTOTYPE OUTPUT
// ============================================================

function findPrototypeOutput(
    outputs
) {

    return outputs.find(
        output => {

            if (
                !output ||
                !output.dims
            ) {
                return false;
            }


            const d =
                output.dims;


            return (
                d.length === 4 &&
                d[0] === 1 &&
                d[1] >= 4 &&
                d[2] >= 8 &&
                d[3] >= 8
            );
        }
    ) || null;
}


// ============================================================
// POSTPROCESS SEGMENTATION
// ============================================================

function postprocessSegmentation(
    detectionOutput,
    prototypeOutput,
    scale,
    offsetX,
    offsetY
) {

    const detDims =
        detectionOutput.dims;

    const detData =
        detectionOutput.data;


    // ========================================================
    // YOLO END-TO-END SEGMENTATION
    //
    // [1, 300, 38]
    //
    // mỗi detection:
    // [x1, y1, x2, y2, confidence, class_id, 32 mask_coeff]
    // ========================================================

    const numBoxes =
        detDims[1];

    const numChannels =
        detDims[2];

    const numClasses =
        CLASS_NAMES.length;


    const protoDims =
        prototypeOutput.dims;

    const protoData =
        prototypeOutput.data;


    const protoChannels =
        protoDims[1];

    const protoHeight =
        protoDims[2];

    const protoWidth =
        protoDims[3];


    console.log(
        "Detection shape:",
        detDims
    );

    console.log(
        "Prototype shape:",
        protoDims
    );

    console.log(
        "Num boxes:",
        numBoxes
    );

    console.log(
        "Num channels:",
        numChannels
    );

    console.log(
        "Prototype channels:",
        protoChannels
    );


    // ========================================================
    // KIỂM TRA FORMAT
    // ========================================================

    if (
        numChannels !==
        6 + protoChannels
    ) {

        throw new Error(
            `Output không đúng format segmentation. ` +
            `Detection=${numChannels}, ` +
            `Expected=${6 + protoChannels}`
        );
    }


    const candidates = [];


    // ========================================================
    // READ 300 DETECTIONS
    // ========================================================

    for (
        let i = 0;
        i < numBoxes;
        i++
    ) {

        const base =
            i *
            numChannels;


        // ----------------------------------------------------
        // BOX
        // ----------------------------------------------------

        let x1 =
            detData[base + 0];

        let y1 =
            detData[base + 1];

        let x2 =
            detData[base + 2];

        let y2 =
            detData[base + 3];


        // ----------------------------------------------------
        // CONFIDENCE
        // ----------------------------------------------------

        const score =
            detData[base + 4];


        // ----------------------------------------------------
        // CLASS ID
        // ----------------------------------------------------

        const classId =
            Math.round(
                detData[base + 5]
            );


        // ----------------------------------------------------
        // CONFIDENCE FILTER
        // ----------------------------------------------------

        if (
            score <
            CONF_THRESHOLD
        ) {
            continue;
        }


        // ----------------------------------------------------
        // CLASS CHECK
        // ----------------------------------------------------

        if (
            classId < 0 ||
            classId >= CLASS_NAMES.length
        ) {
            continue;
        }


        // ====================================================
        // XYXY -> CAMERA COORDINATES
        // ====================================================

        x1 =
            (x1 - offsetX) /
            scale;

        y1 =
            (y1 - offsetY) /
            scale;

        x2 =
            (x2 - offsetX) /
            scale;

        y2 =
            (y2 - offsetY) /
            scale;


        // ====================================================
        // CLAMP
        // ====================================================

        x1 =
            Math.max(
                0,
                Math.min(
                    video.videoWidth,
                    x1
                )
            );

        y1 =
            Math.max(
                0,
                Math.min(
                    video.videoHeight,
                    y1
                )
            );

        x2 =
            Math.max(
                0,
                Math.min(
                    video.videoWidth,
                    x2
                )
            );

        y2 =
            Math.max(
                0,
                Math.min(
                    video.videoHeight,
                    y2
                )
            );


        if (
            x2 <= x1 ||
            y2 <= y1
        ) {
            continue;
        }


        // ====================================================
        // MASK COEFFICIENTS
        //
        // index 6 -> 37
        // tổng cộng 32 coefficients
        // ====================================================

        const maskCoefficients =
            new Float32Array(
                protoChannels
            );


        for (
            let m = 0;
            m < protoChannels;
            m++
        ) {

            maskCoefficients[m] =
                detData[
                    base +
                    6 +
                    m
                ];
        }


        // ====================================================
        // SAVE
        // ====================================================

        candidates.push({

            x1,
            y1,
            x2,
            y2,

            score,

            classId,

            maskCoefficients,

            protoData,

            protoWidth,

            protoHeight,

            protoChannels
        });
    }


    // ========================================================
    // NMS
    // ========================================================
    //
    // Output [1,300,38] đã được Ultralytics xử lý detection.
    // Tuy nhiên vẫn có thể giữ NMS để an toàn.
    //
    // ========================================================

    const selected =
        nms(
            candidates,
            IOU_THRESHOLD
        );


    // ========================================================
    // CREATE MASK
    // ========================================================

    for (
        const detection
        of selected
    ) {

        detection.mask =
            createMask(
                detection
            );
    }


    // ========================================================
    // CHỈ GIỮ DETECTION CUỐI
    // ========================================================

    if (
        selected.length > 0
    ) {

        return [
            selected[
                selected.length - 1
            ]
        ];
    }


    return [];
}


// ============================================================
// CREATE MASK
// ============================================================

function createMask(detection) {

    const {
        maskCoefficients,
        protoData,
        protoWidth,
        protoHeight,
        protoChannels
    } = detection;


    const maskSize =
        protoWidth *
        protoHeight;


    const mask =
        new Uint8Array(
            maskSize
        );


    for (
        let py = 0;
        py < protoHeight;
        py++
    ) {

        for (
            let px = 0;
            px < protoWidth;
            px++
        ) {

            const p =
                py *
                protoWidth +
                px;


            let value = 0;


            for (
                let c = 0;
                c < protoChannels;
                c++
            ) {

                const protoIndex =
                    c *
                    maskSize +
                    p;


                value +=
                    maskCoefficients[c] *
                    protoData[protoIndex];
            }


            const probability =
                1 /
                (
                    1 +
                    Math.exp(-value)
                );


            mask[p] =
                probability >=
                MASK_THRESHOLD
                    ? 255
                    : 0;
        }
    }


    return {

        data:
            mask,

        width:
            protoWidth,

        height:
            protoHeight
    };
}


// ============================================================
// IOU
// ============================================================

function calculateIoU(
    a,
    b
) {

    const x1 =
        Math.max(
            a.x1,
            b.x1
        );

    const y1 =
        Math.max(
            a.y1,
            b.y1
        );

    const x2 =
        Math.min(
            a.x2,
            b.x2
        );

    const y2 =
        Math.min(
            a.y2,
            b.y2
        );


    const intersectionWidth =
        Math.max(
            0,
            x2 - x1
        );


    const intersectionHeight =
        Math.max(
            0,
            y2 - y1
        );


    const intersection =
        intersectionWidth *
        intersectionHeight;


    const areaA =
        Math.max(
            0,
            a.x2 - a.x1
        ) *
        Math.max(
            0,
            a.y2 - a.y1
        );


    const areaB =
        Math.max(
            0,
            b.x2 - b.x1
        ) *
        Math.max(
            0,
            b.y2 - b.y1
        );


    const union =
        areaA +
        areaB -
        intersection;


    if (
        union <= 0
    ) {
        return 0;
    }


    return (
        intersection /
        union
    );
}


// ============================================================
// NMS
// ============================================================

function nms(
    boxes,
    threshold
) {

    const sorted =
        [...boxes].sort(
            (
                a,
                b
            ) =>
                b.score -
                a.score
        );


    const selected = [];


    while (
        sorted.length > 0
    ) {

        const current =
            sorted.shift();


        selected.push(
            current
        );


        for (
            let i =
                sorted.length - 1;

            i >= 0;

            i--
        ) {

            const box =
                sorted[i];


            // NMS theo class
            if (
                box.classId !==
                current.classId
            ) {
                continue;
            }


            const iou =
                calculateIoU(
                    current,
                    box
                );


            if (
                iou >=
                threshold
            ) {

                sorted.splice(
                    i,
                    1
                );
            }
        }
    }


    return selected;
}


// ============================================================
// DRAW SEGMENTATION MASK ONLY
// ============================================================

function drawDetections(detections) {

    overlayCtx.clearRect(
        0,
        0,
        overlay.width,
        overlay.height
    );

    const scaleX =
        overlay.width / video.videoWidth;

    const scaleY =
        overlay.height / video.videoHeight;

    for (const detection of detections) {

        drawMask(
            detection,
            scaleX,
            scaleY
        );
    }

    updateDetectionList(detections);

    objectsElement.textContent =
        detections.length;
}


// ============================================================
// DRAW MASK
// ============================================================

function drawMask(
    detection,
    scaleX,
    scaleY
) {

    const mask =
        detection.mask;


    if (!mask) {
        return;
    }


    const color =
        CLASS_COLORS[
            detection.classId
        ] ||
        [0, 255, 0];


    const maskCanvas =
        document.createElement(
            "canvas"
        );


    maskCanvas.width =
        mask.width;


    maskCanvas.height =
        mask.height;


    const maskCtx =
        maskCanvas.getContext(
            "2d"
        );


    const imageData =
        maskCtx.createImageData(
            mask.width,
            mask.height
        );


    // --------------------------------------------------------
    // MASK PIXELS
    // --------------------------------------------------------

    for (
        let i = 0;
        i < mask.data.length;
        i++
    ) {

        const p =
            i * 4;


        if (
            mask.data[i] > 0
        ) {

            imageData.data[p] =
                color[0];

            imageData.data[p + 1] =
                color[1];

            imageData.data[p + 2] =
                color[2];

            // Transparency
            imageData.data[p + 3] =
                105;
        }

        else {

            imageData.data[p] =
                0;

            imageData.data[p + 1] =
                0;

            imageData.data[p + 2] =
                0;

            imageData.data[p + 3] =
                0;
        }
    }


    maskCtx.putImageData(
        imageData,
        0,
        0
    );


    // --------------------------------------------------------
    // MAP MASK BACK TO CAMERA
    // --------------------------------------------------------

    const videoWidth =
        video.videoWidth;


    const videoHeight =
        video.videoHeight;


    const modelScale =
        Math.min(
            INPUT_SIZE /
            videoWidth,

            INPUT_SIZE /
            videoHeight
        );


    const newWidth =
        videoWidth *
        modelScale;


    const newHeight =
        videoHeight *
        modelScale;


    const modelOffsetX =
        (
            INPUT_SIZE -
            newWidth
        ) / 2;


    const modelOffsetY =
        (
            INPUT_SIZE -
            newHeight
        ) / 2;


    const sourceX =
        modelOffsetX /
        INPUT_SIZE *
        mask.width;


    const sourceY =
        modelOffsetY /
        INPUT_SIZE *
        mask.height;


    const sourceW =
        newWidth /
        INPUT_SIZE *
        mask.width;


    const sourceH =
        newHeight /
        INPUT_SIZE *
        mask.height;


    // --------------------------------------------------------
    // ONLY MASK IS DRAWN
    // --------------------------------------------------------

    overlayCtx.drawImage(

        maskCanvas,

        sourceX,
        sourceY,

        sourceW,
        sourceH,

        0,
        0,

        overlay.width,
        overlay.height
    );
    const className =
    CLASS_NAMES[detection.classId] ||
    `Class ${detection.classId}`;

const confidence =
    (detection.score * 100).toFixed(1);

const label =
    `${className} ${confidence}%`;

overlayCtx.font = "bold 18px Arial";
overlayCtx.fillStyle = "#00ff00";

overlayCtx.fillText(
    label,
    detection.x1 * scaleX,
    Math.max(
        25,
        detection.y1 * scaleY
    )
);
}


// ============================================================
// DETECTION LIST
// ============================================================

function updateDetectionList(
    detections
) {

    if (
        detections.length === 0
    ) {

        detectionsElement.textContent =
            "No detection";

        return;
    }


    detectionsElement.innerHTML =
        detections
            .map(
                detection => {

                    const className =
                        CLASS_NAMES[
                            detection.classId
                        ] ||
                        `Class ${
                            detection.classId
                        }`;


                    const confidence =
                        (
                            detection.score *
                            100
                        ).toFixed(1);


                    const color =
                        CLASS_COLORS[
                            detection.classId
                        ] ||
                        [0, 255, 0];


                    return `

                        <div style="
                            padding:6px 8px;
                            margin-bottom:5px;
                            border-radius:6px;

                            background:rgba(
                                ${color[0]},
                                ${color[1]},
                                ${color[2]},
                                0.15
                            );
                        ">

                            <strong>
                                ${className}
                            </strong>

                            - ${confidence}%

                            - SEGMENTATION

                        </div>

                    `;
                }
            )
            .join("");
}


// ============================================================
// FPS
// ============================================================

function updateFPS() {

    frameCounter++;


    const now =
        performance.now();


    const elapsed =
        now -
        lastFPSUpdate;


    if (
        elapsed >= 1000
    ) {

        const fps =
            frameCounter *
            1000 /
            elapsed;


        fpsElement.textContent =
            fps.toFixed(1);


        frameCounter =
            0;


        lastFPSUpdate =
            now;
    }
}


// ============================================================
// REALTIME LOOP
// ============================================================

async function inferenceLoop(
    timestamp
) {

    if (!running) {
        return;
    }


    if (
        video.readyState >= 2 &&

        timestamp -
        lastInferenceTime >=
        INFERENCE_INTERVAL
    ) {

        lastInferenceTime =
            timestamp;


        if (!inferenceBusy) {

            inferenceBusy =
                true;


            try {

                const result =
                    await runInference();


                if (result) {

                    const detections =
                        postprocessSegmentation(

                            result.detectionOutput,

                            result.prototypeOutput,

                            result.scale,

                            result.offsetX,

                            result.offsetY
                        );


                    // ------------------------------------------------
                    // DRAW MASK ONLY
                    // ------------------------------------------------

                    drawDetections(
                        detections
                    );


                    updateFPS();


                    if (
                        detections.length >
                        0
                    ) {

                        setStatus(
                            `Segmented ${
                                detections.length
                            } object(s)`
                        );

                    }

                    else {

                        setStatus(
                            "Scanning..."
                        );
                    }
                }

            }

            catch (error) {

                console.error(
                    "SEGMENTATION ERROR:",
                    error
                );


                setStatus(
                    "Segmentation error - see Console"
                );

            }

            finally {

                inferenceBusy =
                    false;
            }
        }
    }


    requestAnimationFrame(
        inferenceLoop
    );
}


// ============================================================
// BUTTON EVENTS
// ============================================================

if (loadButton) {

    loadButton.addEventListener(
        "click",
        loadModel
    );
}


if (cameraButton) {

    cameraButton.addEventListener(
        "click",
        startCamera
    );
}


if (stopButton) {

    stopButton.addEventListener(
        "click",
        stopCamera
    );
}


// ============================================================
// VIDEO METADATA
// ============================================================

video.addEventListener(
    "loadedmetadata",
    () => {

        setupOverlay();

    }
);


// ============================================================
// WINDOW RESIZE
// ============================================================

window.addEventListener(
    "resize",
    () => {

        setupOverlay();

    }
);


// ============================================================
// INITIAL STATUS
// ============================================================

setStatus(
    "Ready - load segmentation model"
);