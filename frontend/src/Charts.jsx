import {
  ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, Legend,
  CartesianGrid, BarChart, Bar, LabelList,
} from "recharts";

const COLORS = { FedAvg: "#2a7fff", FedProx: "#e8590c", FedCurv: "#2f9e44" };

export default function Charts({ compare }) {
  const names = Object.keys(compare || {});
  if (names.length === 0) return <p>Loading comparison results...</p>;

  const maxRounds = Math.max(...names.map((n) => compare[n].rounds.length));
  const lineData = Array.from({ length: maxRounds }, (_, i) => {
    const row = { round: i + 1 };
    names.forEach((n) => {
      row[n] = compare[n].acc[i];
    });
    return row;
  });

  const barData = names.map((n) => ({
    name: n,
    best: Math.max(...compare[n].acc),
    final: compare[n].acc[compare[n].acc.length - 1],
  }));

  return (
    <div>
      <h3>Test accuracy per round (%)</h3>
      <div style={{ height: 280 }}>
        <ResponsiveContainer>
          <LineChart data={lineData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="round" />
            <YAxis domain={[60, 100]} />
            <Tooltip />
            <Legend />
            {names.map((n) => (
              <Line key={n} type="monotone" dataKey={n}
                stroke={COLORS[n] || "#555"} strokeWidth={2} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>

      <h3>Best vs final accuracy (%)</h3>
      <div style={{ height: 280 }}>
        <ResponsiveContainer>
          <BarChart data={barData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis domain={[60, 100]} />
            <Tooltip />
            <Legend />
            <Bar dataKey="best" name="Best round" fill="#2a7fff">
              <LabelList dataKey="best" position="top" />
            </Bar>
            <Bar dataKey="final" name="Final round" fill="#adb5bd">
              <LabelList dataKey="final" position="top" />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}