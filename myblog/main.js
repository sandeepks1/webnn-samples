'use strict';

import {MobileNetV2Nchw} from '../image_classification/mobilenet_nchw.js';
import * as ui from '../common/ui.js';
import * as utils from '../common/utils.js';

const VIDEO_URL = 'https://genuine-marzipan-c4a520.netlify.app/video3.mp4';
const maxWidth = 640;
const maxHeight = 640;

const videoElement = document.getElementById('feedMediaElement');
const canvas = document.getElementById('camInCanvas');
const ctx = canvas.getContext('2d');

let modelName = 'mobilenet';
let layout = 'nchw';
let dataType = 'float32';
let deviceType = 'npu';
let netInstance = null;
let labels = null;
let loadTime = 0;
let buildTime = 0;
let computeTime = 0;
let inputOptions;
let isProcessing = false;

async function fetchLabels(url) {
  const response = await fetch(url);
  const data = await response.text();
  return data.split('\n');
}

async function main() {
  try {
    ui.addAlert('Loading model...', 'info');
    $('#status').html('Loading model...');
    
    // Load labels
    labels = await fetchLabels('../image_classification/labels/labels1001.txt');
    
    // Initialize model
    const contextOptions = {deviceType: deviceType};
    netInstance = new MobileNetV2Nchw();
    
    inputOptions = {
      inputLayout: layout,
      inputDimensions: [1, 3, 224, 224],
    };
    
    const start = performance.now();
    const outputOperand = await netInstance.load(contextOptions);
    loadTime = (performance.now() - start).toFixed(2);
    
    const buildStart = performance.now();
    await netInstance.build(outputOperand);
    buildTime = (performance.now() - buildStart).toFixed(2);
    
    $('#loadTime').html(`${loadTime} ms`);
    $('#buildTime').html(`${buildTime} ms`);
    
    ui.addAlert('Model loaded successfully!', 'success');
    $('#status').html('Model ready. Processing video...');
    
    // Start video
    videoElement.src = VIDEO_URL;
    videoElement.onloadedmetadata = () => {
      canvas.width = videoElement.videoWidth;
      canvas.height = videoElement.videoHeight;
      processFrame();
    };
    
  } catch (error) {
    console.error('Error:', error);
    ui.addAlert(`Error: ${error.message}`, 'danger');
    $('#status').html(`Error: ${error.message}`);
  }
}

async function processFrame() {
  if (!netInstance || !videoElement.paused === false) {
    requestAnimationFrame(processFrame);
    return;
  }
  
  if (isProcessing) {
    requestAnimationFrame(processFrame);
    return;
  }
  
  isProcessing = true;
  
  try {
    // Draw current video frame to canvas
    ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
    
    // Get input tensor from canvas
    const inputBuffer = utils.getInputTensor(canvas, inputOptions);
    
    // Run inference
    const computeStart = performance.now();
    const outputs = await netInstance.compute(inputBuffer);
    computeTime = (performance.now() - computeStart).toFixed(2);
    
    $('#computeTime').html(`${computeTime} ms`);
    
    // Get top 3 predictions
    const outputData = outputs.output;
    const probs = Array.from(outputData);
    const indexes = probs.map((prob, index) => [prob, index]);
    const sorted = indexes.sort((a, b) => {
      if (a[0] === b[0]) return 0;
      return a[0] < b[0] ? 1 : -1;
    });
    sorted.splice(3);
    
    // Display results
    for (let i = 0; i < 3; i++) {
      const prob = sorted[i][0];
      const index = sorted[i][1];
      const label = labels[index];
      $(`#label${i}`).html(label);
      $(`#prob${i}`).html(`${(prob * 100).toFixed(2)}%`);
    }
    
  } catch (error) {
    console.error('Processing error:', error);
  }
  
  isProcessing = false;
  requestAnimationFrame(processFrame);
}

// Auto-start on page load
$(document).ready(async () => {
  if (await utils.isWebNN()) {
    $('#npu').prop('checked', true).parent().addClass('active');
    $('#mobilenet').prop('checked', true).parent().addClass('active');
    await main();
  } else {
    console.log(utils.webNNNotSupportMessage());
    ui.addAlert(utils.webNNNotSupportMessageHTML(), 'danger');
    $('#status').html('WebNN not supported');
  }
});
