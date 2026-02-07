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