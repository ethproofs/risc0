const express = require("express");
const { Pool } = require("pg");
const cors = require("cors");
const path = require("path");

const app = express();
const PORT = process.env.PORT || 3001;

// PostgreSQL connection configuration
const pool = new Pool({
    host: process.env.POSTGRES_HOST || 'postgres',
    port: process.env.POSTGRES_PORT || 5432,
    database: process.env.POSTGRES_DB || 'taskdb',
    user: process.env.POSTGRES_USER || 'worker',
    password: process.env.POSTGRES_PASSWORD || 'password',
});

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

// Test database connection
pool.query('SELECT NOW()', (err, res) => {
    if (err) {
        console.error('Database connection error:', err);
    } else {
        console.log('Database connected successfully');
    }
});

// Get all jobs with task statistics
app.get("/api/jobs", async (req, res) => {
    try {
        const query = `
            SELECT
                j.id,
                j.state,
                j.error,
                j.user_id,
                j.reported,
                COUNT(t.task_id) as task_count,
                COUNT(CASE WHEN t.state = 'done' THEN 1 END) as completed_tasks,
                COUNT(CASE WHEN t.state = 'running' THEN 1 END) as running_tasks,
                COUNT(CASE WHEN t.state = 'pending' THEN 1 END) as pending_tasks,
                COUNT(CASE WHEN t.state = 'failed' THEN 1 END) as failed_tasks
            FROM jobs j
            LEFT JOIN tasks t ON j.id = t.job_id
            GROUP BY j.id, j.state, j.error, j.user_id, j.reported
            ORDER BY j.id DESC
        `;

        const result = await pool.query(query);
        res.json(result.rows);
    } catch (error) {
        console.error("Error fetching jobs:", error);
        res.status(500).json({ error: "Failed to fetch jobs" });
    }
});

// Get job details with all tasks
app.get("/api/jobs/:jobId", async (req, res) => {
    try {
        const { jobId } = req.params;

        // Get job details
        const jobQuery = `
            SELECT
                id,
                state,
                error,
                user_id,
                reported
            FROM jobs
            WHERE id = $1
        `;

        const jobResult = await pool.query(jobQuery, [jobId]);

        if (jobResult.rows.length === 0) {
            return res.status(404).json({ error: "Job not found" });
        }

        const job = jobResult.rows[0];

        // Get tasks for this job
        const tasksQuery = `
            SELECT
                task_id,
                state,
                progress,
                retries,
                max_retries,
                timeout_secs,
                waiting_on,
                error,
                created_at,
                started_at,
                updated_at,
                task_def,
                prerequisites,
                output
            FROM tasks
            WHERE job_id = $1
            ORDER BY created_at ASC
        `;

        const tasksResult = await pool.query(tasksQuery, [jobId]);

        res.json({
            job: job,
            tasks: tasksResult.rows
        });
    } catch (error) {
        console.error("Error fetching job details:", error);
        res.status(500).json({ error: "Failed to fetch job details" });
    }
});

// Get task dependencies
app.get("/api/jobs/:jobId/dependencies", async (req, res) => {
    try {
        const { jobId } = req.params;

        const query = `
            SELECT
                pre_task_id,
                post_task_id
            FROM task_deps
            WHERE job_id = $1
        `;

        const result = await pool.query(query, [jobId]);
        res.json(result.rows);
    } catch (error) {
        console.error("Error fetching task dependencies:", error);
        res.status(500).json({ error: "Failed to fetch task dependencies" });
    }
});

// Get streams
app.get("/api/streams", async (req, res) => {
    try {
        const query = `
            SELECT
                id,
                job_id,
                created_at,
                updated_at
            FROM streams
            ORDER BY created_at DESC
        `;

        const result = await pool.query(query);
        res.json(result.rows);
    } catch (error) {
        console.error("Error fetching streams:", error);
        res.status(500).json({ error: "Failed to fetch streams" });
    }
});

// Serve the main HTML page
app.get("/", (req, res) => {
    res.sendFile(path.join(__dirname, "public", "index.html"));
});

// Health check endpoint
app.get("/health", (req, res) => {
    res.json({
        status: "ok",
        timestamp: new Date().toISOString(),
        database: "connected"
    });
});

app.listen(PORT, () => {
    console.log(`Bento DB Web App running on http://localhost:${PORT}`);
    console.log(`Database: ${process.env.POSTGRES_HOST || 'postgres'}:${process.env.POSTGRES_PORT || 5432}/${process.env.POSTGRES_DB || 'taskdb'}`);
});

// Graceful shutdown
process.on('SIGINT', () => {
    console.log('Shutting down gracefully...');
    pool.end();
    process.exit(0);
});
