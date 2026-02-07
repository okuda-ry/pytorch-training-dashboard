let lossChart, accChart;
let epochCount = 0;

window.onload = () => {
 const lossCtx = document.getElementById('lossChart');
 const accCtx = document.getElementById('accChart');

 lossChart = new Chart(lossCtx, {
  type: 'line',
  data: {
   labels: [],
   datasets: [{
    label: 'Loss',
    data: [],
    borderColor: '#ff6b6b',
    backgroundColor: 'rgba(255, 107, 107, 0.1)',
    tension: 0.3,
    fill: true
   }]
  },
  options: {
   animation: false,
   responsive: true,
   plugins: {
    filler: {
     propagate: true
    }
   },
   scales: {
    y: {
     beginAtZero: true,
     max: 1.0
    }
   }
  },
  plugins: [{
   id: 'customCanvasBackgroundColor',
   beforeDraw(chart, args, options) {
    const {ctx} = chart;
    ctx.save();
    ctx.globalCompositeOperation = 'destination-over';
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, chart.width, chart.height);
    ctx.restore();
   }
  }]
 });

 accChart = new Chart(accCtx, {
  type: 'line',
  data: {
   labels: [],
   datasets: [{
    label: 'Accuracy',
    data: [],
    borderColor: '#00ffc3',
    backgroundColor: 'rgba(0, 255, 195, 0.1)',
    tension: 0.3,
    fill: true
   }]
  },
  options: {
   animation: false,
   responsive: true,
   plugins: {
    filler: {
     propagate: true
    }
   },
   scales: {
    y: {
     beginAtZero: true,
     max: 1.0
    }
   }
  },
  plugins: [{
   id: 'customCanvasBackgroundColor',
   beforeDraw(chart, args, options) {
    const {ctx} = chart;
    ctx.save();
    ctx.globalCompositeOperation = 'destination-over';
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, chart.width, chart.height);
    ctx.restore();
   }
  }]
 });
}

async function startTraining(){
 const button = event.target;
 const epochInput = document.getElementById('epochInput');
 const epochs = parseInt(epochInput.value) || 10;
 
 button.disabled = true;
 button.textContent = 'Training...';
 epochInput.disabled = true;

 try {
  const res = await fetch("http://127.0.0.1:5000/train", {
   method: "POST",
   headers: {
    "Content-Type": "application/json"
   },
   body: JSON.stringify({ epochs: epochs })
  });
  const data = await res.json();

  console.log(data);

  // Clear existing data
  lossChart.data.labels = [];
  lossChart.data.datasets[0].data = [];
  accChart.data.labels = [];
  accChart.data.datasets[0].data = [];

  // Add data points with animation
  for (let i = 0; i < data.loss.length; i++) {
   await new Promise(resolve => setTimeout(resolve, 100));
   
   lossChart.data.labels.push('Epoch ' + (i + 1));
   lossChart.data.datasets[0].data.push(data.loss[i]);
   lossChart.update();

   accChart.data.labels.push('Epoch ' + (i + 1));
   accChart.data.datasets[0].data.push(data.acc[i]);
   accChart.update();
  }

  console.log('Training completed!');
 } catch (error) {
  console.error('Error:', error);
  alert('Error during training: ' + error);
 } finally {
  button.disabled = false;
  button.textContent = 'Train Model';
  epochInput.disabled = false;
 }
}

// Drawing Canvas Setup
const canvas = document.getElementById('drawingCanvas');
const ctx = canvas.getContext('2d');
let isDrawing = false;

// Set canvas background to white
ctx.fillStyle = 'white';
ctx.fillRect(0, 0, canvas.width, canvas.height);

canvas.addEventListener('mousedown', (e) => {
  isDrawing = true;
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  ctx.beginPath();
  ctx.moveTo(x, y);
});

canvas.addEventListener('mousemove', (e) => {
  if (!isDrawing) return;
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const y = e.clientY - rect.top;
  ctx.lineWidth = 25;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = 'black';
  ctx.lineTo(x, y);
  ctx.stroke();
});

canvas.addEventListener('mouseup', () => {
  isDrawing = false;
});

canvas.addEventListener('mouseout', () => {
  isDrawing = false;
});

// Clear Canvas Button
document.getElementById('clearBtn').addEventListener('click', () => {
  ctx.fillStyle = 'white';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  document.getElementById('predictionResult').classList.add('hidden');
});

// Predict Button
document.getElementById('predictBtn').addEventListener('click', () => {
  const imageData = canvas.toDataURL('image/png');
  fetch('/predict', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ image: imageData }),
  })
    .then((response) => response.json())
    .then((data) => {
      document.getElementById('predictedDigit').textContent = data.digit;
      document.getElementById('confidence').textContent = (data.confidence * 100).toFixed(2) + '%';
      document.getElementById('predictionResult').classList.remove('hidden');
    })
    .catch((error) => console.error('Error:', error));
});