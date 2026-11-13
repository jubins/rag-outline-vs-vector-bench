import http from 'k6/http';
import { check } from 'k6';

const BASE = __ENV.BASE_URL || 'http://orchestrator:8000';
const DESIGN = __ENV.DESIGN || 'outline';

export const options = {
  scenarios: {
    steady: {
      executor: 'constant-arrival-rate',
      rate: Number(__ENV.RATE || 5),
      timeUnit: '1s',
      duration: __ENV.DURATION || '5m',
      preAllocatedVUs: 20,
      maxVUs: 200,
    },
  },
  thresholds: {
    // The latency target used in the article. The run fails if P99 goes above it.
    http_req_duration: ['p(99)<8000'],
  },
};

export function setup() {
  const response = http.get(`${BASE}/questions?n=${__ENV.QUESTIONS || 1000}`);
  return response.json();
}

export default function (questions) {
  const q = questions[Math.floor(Math.random() * questions.length)];
  const response = http.post(
    `${BASE}/ask`,
    JSON.stringify({ question: q.question, doc_id: q.doc_id, design: DESIGN }),
    { headers: { 'Content-Type': 'application/json' }, timeout: '60s' },
  );
  check(response, { 'status is 200': (r) => r.status === 200 });
}
