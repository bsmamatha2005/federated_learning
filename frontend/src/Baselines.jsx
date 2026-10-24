import {
  ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Legend,
  CartesianGrid, LabelList, Cell,
} from "recharts";

const COLORS = {
  Centralized: "#444444", "Local-only": "#adb5bd",
  FedAvg: "#2a7fff", FedProx: "#e8590c", FedCurv: "#2f9e44",
};

export default function Baselines({ data }) {
  if (!data || !data.hospitals) return <p>Loading baseline results...</p>;

  const models = [
    ["Centralized", data.centralized],
    ["Local-only", data.local],
    ...Object.entries(data.federated || {}),
  ];

  const overall = models.map(([name, d]) => ({ name, accuracy: d.overall }));

  const perHospital = data.hospitals.map((h, i) => {
    const row = { hospital: `${h} (n=${data.test_sizes[i]})` };
    models.forEach(([name, d]) => { row[name] = d.per_hospital[i]; });
    return row;
  });

  const metricRows = models.filter(([, d]) => d.precision !== undefined);

  return (
    <div>
      <h3>Centralized vs Local-only vs Federated (overall accuracy %)</h3>
      <div style={{ height: 300 }}>
        <ResponsiveContainer>
          <BarChart data={overall}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis domain={[60, 100]} />
            <Tooltip />
            <Bar dataKey="accuracy" name="Accuracy %">
              {overall.map((d) => (
                <Cell key={d.name} fill={COLORS[d.name] || "#555"} />
              ))}
              <LabelList dataKey="accuracy" position="top" />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <h3>Accuracy per hospital (%)</h3>
      <div style={{ height: 340 }}>
        <ResponsiveContainer>
          <BarChart data={perHospital}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="hospital" />
            <YAxis domain={[50, 100]} />
            <Tooltip />
            <Legend />
            {models.map(([name]) => (
              <Bar key={name} dataKey={name} fill={COLORS[name] || "#555"} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p style={{ fontSize: 13, color: "#555" }}>
        Hospital test sets are small, so one patient changes a bar by about
        2 to 4%. Switzerland is about 93% positive, which inflates local-only accuracy there.
      </p>

      <h3>Medical metrics (pooled test set, %)</h3>
      <table border="1" cellPadding="6"
        style={{ width: "100%", borderCollapse: "collapse" }}>
        <thead>
          <tr><th>Model</th><th>Accuracy</th><th>Precision</th>
            <th>Recall</th><th>F1</th><th>ROC-AUC</th></tr>
        </thead>
        <tbody>
          {metricRows.map(([name, d]) => (
            <tr key={name}>
              <td>{name}</td><td>{d.overall}</td><td>{d.precision}</td>
              <td>{d.recall}</td><td>{d.f1}</td><td>{d.roc_auc}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}