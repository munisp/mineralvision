import http from 'k6/http';
import { check, sleep, group } from 'k6';
import { Rate, Trend, Counter } from 'k6/metrics';
import { randomString, randomIntBetween } from 'https://jslib.k6.io/k6-utils/1.2.0/index.js';

// Custom metrics
const errorRate = new Rate('errors');
const sensorFusionDuration = new Trend('sensor_fusion_duration');
const predictionDuration = new Trend('prediction_duration');
const apiRequests = new Counter('api_requests');

// Test configuration
export const options = {
  scenarios: {
    // Smoke test - verify system works
    smoke: {
      executor: 'constant-vus',
      vus: 1,
      duration: '1m',
      startTime: '0s',
      tags: { test_type: 'smoke' },
    },
    // Load test - normal expected load
    load: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 50 },   // Ramp up
        { duration: '5m', target: 50 },   // Stay at 50 users
        { duration: '2m', target: 100 },  // Ramp up to 100
        { duration: '5m', target: 100 },  // Stay at 100 users
        { duration: '2m', target: 0 },    // Ramp down
      ],
      startTime: '1m',
      tags: { test_type: 'load' },
    },
    // Stress test - beyond normal capacity
    stress: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '2m', target: 100 },
        { duration: '5m', target: 200 },
        { duration: '5m', target: 300 },
        { duration: '5m', target: 400 },
        { duration: '2m', target: 0 },
      ],
      startTime: '17m',
      tags: { test_type: 'stress' },
    },
    // Spike test - sudden traffic spike
    spike: {
      executor: 'ramping-vus',
      startVUs: 0,
      stages: [
        { duration: '10s', target: 500 },  // Spike to 500 users
        { duration: '1m', target: 500 },   // Stay at 500
        { duration: '10s', target: 0 },    // Drop to 0
      ],
      startTime: '36m',
      tags: { test_type: 'spike' },
    },
    // Soak test - extended duration
    soak: {
      executor: 'constant-vus',
      vus: 50,
      duration: '30m',
      startTime: '38m',
      tags: { test_type: 'soak' },
    },
  },
  thresholds: {
    http_req_duration: ['p(95)<500', 'p(99)<1000'],
    http_req_failed: ['rate<0.01'],
    errors: ['rate<0.05'],
    sensor_fusion_duration: ['p(95)<5000'],
    prediction_duration: ['p(95)<3000'],
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:8000';
const API_KEY = __ENV.API_KEY || 'test-api-key';

const headers = {
  'Content-Type': 'application/json',
  'Authorization': `Bearer ${API_KEY}`,
};

// Helper function to generate test data
function generateSensorData() {
  return {
    sensor_type: ['hyperspectral', 'lidar', 'magnetometry'][randomIntBetween(0, 2)],
    data: {
      values: Array.from({ length: 100 }, () => Math.random() * 1000),
      coordinates: {
        lat: -23.5 + Math.random() * 10,
        lon: 119.5 + Math.random() * 10,
      },
      timestamp: new Date().toISOString(),
    },
    metadata: {
      source: `sensor-${randomString(8)}`,
      quality: Math.random(),
    },
  };
}

function generatePredictionRequest() {
  return {
    region: {
      min_lon: 119.0,
      max_lon: 120.0,
      min_lat: -24.0,
      max_lat: -23.0,
    },
    mineral_types: ['gold', 'iron', 'copper'][randomIntBetween(0, 2)],
    confidence_threshold: 0.7,
  };
}

export default function () {
  group('Health Check', function () {
    const res = http.get(`${BASE_URL}/health`, { headers });
    check(res, {
      'health check status is 200': (r) => r.status === 200,
      'health check response time < 100ms': (r) => r.timings.duration < 100,
    });
    apiRequests.add(1);
    errorRate.add(res.status !== 200);
  });

  group('Sensor Fusion API', function () {
    // Upload sensor data
    const sensorData = generateSensorData();
    const uploadRes = http.post(
      `${BASE_URL}/api/sensor-fusion/upload`,
      JSON.stringify(sensorData),
      { headers }
    );
    check(uploadRes, {
      'sensor upload status is 200 or 201': (r) => r.status === 200 || r.status === 201,
    });
    apiRequests.add(1);
    errorRate.add(uploadRes.status !== 200 && uploadRes.status !== 201);

    if (uploadRes.status === 200 || uploadRes.status === 201) {
      const uploadData = JSON.parse(uploadRes.body);
      const dataId = uploadData.data_id || uploadData.id;

      // Trigger fusion
      const fusionStart = Date.now();
      const fusionRes = http.post(
        `${BASE_URL}/api/sensor-fusion/fuse`,
        JSON.stringify({
          data_ids: [dataId],
          algorithm: 'bayesian',
        }),
        { headers }
      );
      sensorFusionDuration.add(Date.now() - fusionStart);
      check(fusionRes, {
        'fusion status is 200 or 202': (r) => r.status === 200 || r.status === 202,
      });
      apiRequests.add(1);
      errorRate.add(fusionRes.status !== 200 && fusionRes.status !== 202);
    }
  });

  group('Predictive Modeling API', function () {
    const predictionReq = generatePredictionRequest();
    const predStart = Date.now();
    const predRes = http.post(
      `${BASE_URL}/api/predictive-modeling/predict`,
      JSON.stringify(predictionReq),
      { headers }
    );
    predictionDuration.add(Date.now() - predStart);
    check(predRes, {
      'prediction status is 200': (r) => r.status === 200,
      'prediction has results': (r) => {
        try {
          const body = JSON.parse(r.body);
          return body.predictions !== undefined || body.results !== undefined;
        } catch {
          return false;
        }
      },
    });
    apiRequests.add(1);
    errorRate.add(predRes.status !== 200);
  });

  group('Digital Twin API', function () {
    // List entities
    const listRes = http.get(`${BASE_URL}/api/digital-twin/entities`, { headers });
    check(listRes, {
      'list entities status is 200': (r) => r.status === 200,
    });
    apiRequests.add(1);
    errorRate.add(listRes.status !== 200);

    // Create entity
    const createRes = http.post(
      `${BASE_URL}/api/digital-twin/entities`,
      JSON.stringify({
        name: `test-entity-${randomString(8)}`,
        type: 'exploration_area',
        properties: {
          area_km2: randomIntBetween(10, 1000),
          status: 'active',
        },
      }),
      { headers }
    );
    check(createRes, {
      'create entity status is 200 or 201': (r) => r.status === 200 || r.status === 201,
    });
    apiRequests.add(1);
    errorRate.add(createRes.status !== 200 && createRes.status !== 201);
  });

  group('Climate Resilience API', function () {
    const climateRes = http.post(
      `${BASE_URL}/api/climate-resilience/analyze`,
      JSON.stringify({
        region: {
          min_lon: 119.0,
          max_lon: 120.0,
          min_lat: -24.0,
          max_lat: -23.0,
        },
        analysis_type: 'extreme_weather',
        time_range: ['2024-01-01', '2024-12-31'],
      }),
      { headers }
    );
    check(climateRes, {
      'climate analysis status is 200': (r) => r.status === 200,
    });
    apiRequests.add(1);
    errorRate.add(climateRes.status !== 200);
  });

  group('Blockchain Provenance API', function () {
    const blockchainRes = http.post(
      `${BASE_URL}/api/blockchain/register`,
      JSON.stringify({
        data_type: 'sensor_reading',
        metadata: {
          source: 'load-test',
          timestamp: new Date().toISOString(),
        },
        offline_mode: true,
      }),
      { headers }
    );
    check(blockchainRes, {
      'blockchain register status is 200 or 201': (r) => r.status === 200 || r.status === 201,
    });
    apiRequests.add(1);
    errorRate.add(blockchainRes.status !== 200 && blockchainRes.status !== 201);
  });

  sleep(randomIntBetween(1, 3));
}

export function handleSummary(data) {
  return {
    'load-test-results.json': JSON.stringify(data, null, 2),
    'load-test-summary.html': htmlReport(data),
    stdout: textSummary(data, { indent: ' ', enableColors: true }),
  };
}

function textSummary(data, options) {
  const indent = options.indent || '';
  let summary = '\n' + indent + '=== LOAD TEST SUMMARY ===\n\n';
  
  summary += indent + 'Scenarios:\n';
  for (const [name, scenario] of Object.entries(data.root_group.groups)) {
    summary += indent + `  ${name}: ${scenario.checks.passes}/${scenario.checks.passes + scenario.checks.fails} checks passed\n`;
  }
  
  summary += '\n' + indent + 'Metrics:\n';
  summary += indent + `  Total Requests: ${data.metrics.http_reqs.values.count}\n`;
  summary += indent + `  Request Duration (p95): ${data.metrics.http_req_duration.values['p(95)'].toFixed(2)}ms\n`;
  summary += indent + `  Error Rate: ${(data.metrics.http_req_failed.values.rate * 100).toFixed(2)}%\n`;
  
  return summary;
}

function htmlReport(data) {
  return `
<!DOCTYPE html>
<html>
<head>
  <title>MineralVision Load Test Report</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; }
    h1 { color: #333; }
    .metric { margin: 10px 0; padding: 10px; background: #f5f5f5; border-radius: 4px; }
    .pass { color: green; }
    .fail { color: red; }
    table { border-collapse: collapse; width: 100%; }
    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
    th { background-color: #4CAF50; color: white; }
  </style>
</head>
<body>
  <h1>MineralVision Load Test Report</h1>
  <p>Generated: ${new Date().toISOString()}</p>
  
  <h2>Summary</h2>
  <div class="metric">
    <strong>Total Requests:</strong> ${data.metrics.http_reqs.values.count}
  </div>
  <div class="metric">
    <strong>Request Duration (p95):</strong> ${data.metrics.http_req_duration.values['p(95)'].toFixed(2)}ms
  </div>
  <div class="metric">
    <strong>Error Rate:</strong> <span class="${data.metrics.http_req_failed.values.rate < 0.01 ? 'pass' : 'fail'}">${(data.metrics.http_req_failed.values.rate * 100).toFixed(2)}%</span>
  </div>
  
  <h2>Thresholds</h2>
  <table>
    <tr><th>Metric</th><th>Threshold</th><th>Value</th><th>Status</th></tr>
    <tr>
      <td>http_req_duration (p95)</td>
      <td>&lt; 500ms</td>
      <td>${data.metrics.http_req_duration.values['p(95)'].toFixed(2)}ms</td>
      <td class="${data.metrics.http_req_duration.values['p(95)'] < 500 ? 'pass' : 'fail'}">${data.metrics.http_req_duration.values['p(95)'] < 500 ? 'PASS' : 'FAIL'}</td>
    </tr>
    <tr>
      <td>http_req_failed</td>
      <td>&lt; 1%</td>
      <td>${(data.metrics.http_req_failed.values.rate * 100).toFixed(2)}%</td>
      <td class="${data.metrics.http_req_failed.values.rate < 0.01 ? 'pass' : 'fail'}">${data.metrics.http_req_failed.values.rate < 0.01 ? 'PASS' : 'FAIL'}</td>
    </tr>
  </table>
</body>
</html>
  `;
}
