import http from 'k6/http';
import { check } from 'k6';
import { SharedArray } from 'k6/data';

const cards = new SharedArray('cards', function () {
    const f = open('./cards.txt');
    return f.split('\n').filter(line => line.trim().length > 0);
});

export const options = {
    scenarios: {
        drops: {
            executor: 'constant-vus',
            vus: 10,
            duration: '10s',
        },
    },
    thresholds: {
        http_req_duration: ['p(95)<500'],
        http_req_failed: ['rate<0.01'],
    },
};

function randomInt(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
}

export default function () {
    const leftCard = cards[randomInt(0, cards.length - 1)];
    const rightCard = cards[randomInt(0, cards.length - 1)];
    
    const leftPrint = randomInt(1, 999);
    const rightPrint = randomInt(1, 999);

    const baseUrl = __ENV.BASE_URL || 'http://localhost:3000';
    const url = `${baseUrl}/render/drop?left=${encodeURIComponent(leftCard)}&right=${encodeURIComponent(rightCard)}&left_print=${leftPrint}&right_print=${rightPrint}`;

    const res = http.get(url);

    check(res, {
        'status is 200': (r) => r.status === 200,
        'has non-empty body': (r) => r.body.length > 0,
    });
}
