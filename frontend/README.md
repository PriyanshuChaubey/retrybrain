# Frontend (Day 8)

The dashboard is scaffolded on Day 8. Recommended setup:

```bash
npm create vite@latest . -- --template react
npm install
npm install recharts
npm run dev
```

Build these views (see the plan, sections 4 and 13):

- Live batch feed of failed payments
- **₹-recovered counter** (poll `GET /metrics`)
- Recovery-rate vs. baseline chart (Recharts)
- Per-transaction drill-down: root cause → score → decision → agent reasoning → message
- **Audit trail** view
- **Exception list** ("could not recover", with reasons)
