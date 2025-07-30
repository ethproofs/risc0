let jobs = [];
let isConnected = false;
let currentJobDetail = null;
let currentChart = null;

// Update connection status
function updateConnectionStatus(connected, message = '') {
    const statusDot = document.querySelector('.status-dot');
    const statusText = document.querySelector('.status-text');

    isConnected = connected;

    if (connected) {
        statusDot.className = 'status-dot connected';
        statusText.textContent = 'Connected';
    } else {
        statusDot.className = 'status-dot';
        statusText.textContent = message || 'Disconnected';
    }
}

// Check API connection
async function checkConnection() {
    try {
        const response = await fetch('/health');
        if (response.ok) {
            updateConnectionStatus(true);
            return true;
        } else {
            updateConnectionStatus(false, 'API Error');
            return false;
        }
    } catch (error) {
        updateConnectionStatus(false, 'Connection Failed');
        return false;
    }
}

async function loadJobs() {
    try {
        // Check connection first
        const connected = await checkConnection();
        if (!connected) {
            document.getElementById('jobs-container').innerHTML =
                '<div class="error">Cannot connect to Bento API. Please check if the API server is running.</div>';
            return;
        }

        const response = await fetch('/api/jobs');

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        jobs = await response.json();
        displayJobs();
        updateConnectionStatus(true);
    } catch (error) {
        console.error('Error loading jobs:', error);
        updateConnectionStatus(false, 'API Error');
        document.getElementById('jobs-container').innerHTML =
            '<div class="error">Error loading jobs: ' + error.message + '</div>';
    }
}

function displayJobs() {
    const container = document.getElementById('jobs-container');

    if (jobs.length === 0) {
        container.innerHTML = '<div class="loading">No jobs found</div>';
        return;
    }

    container.innerHTML = `
        <div class="jobs-grid">
            ${jobs.map(job => `
                <div class="job-card">
                    <div class="job-header">
                        <div class="job-id">${job.id}</div>
                        <div class="job-state state-${job.state.toLowerCase()}">${job.state}</div>
                    </div>

                    <div class="job-stats">
                        <div class="stat">
                            <div class="stat-value">${job.task_count}</div>
                            <div class="stat-label">Total Tasks</div>
                        </div>
                        <div class="stat">
                            <div class="stat-value">${job.completed_tasks}</div>
                            <div class="stat-label">Completed</div>
                        </div>
                        <div class="stat">
                            <div class="stat-value">${job.running_tasks}</div>
                            <div class="stat-label">Running</div>
                        </div>
                        <div class="stat">
                            <div class="stat-value">${job.pending_tasks}</div>
                            <div class="stat-label">Pending</div>
                        </div>
                        <div class="stat">
                            <div class="stat-value">${job.failed_tasks}</div>
                            <div class="stat-label">Failed</div>
                        </div>
                    </div>

                    ${job.error ? `<div class="job-error">${job.error}</div>` : ''}

                    <button class="view-details-btn" onclick="viewJobDetails('${job.id}')">
                        View Details
                    </button>
                </div>
            `).join('')}
        </div>
    `;
}

async function viewJobDetails(jobId) {
    try {
        if (!isConnected) {
            alert('Not connected to API. Please check the connection.');
            return;
        }

        const response = await fetch(`/api/jobs/${jobId}`);

        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const jobDetail = await response.json();
        currentJobDetail = jobDetail;
        displayJobDetails(jobDetail);
        document.getElementById('jobModal').style.display = 'block';
    } catch (error) {
        console.error('Error loading job details:', error);
        alert('Error loading job details: ' + error.message);
    }
}

function displayJobDetails(jobDetail) {
    const modalContent = document.getElementById('modal-content');
    const job = jobDetail.job;
    const tasks = jobDetail.tasks;

    modalContent.innerHTML = `
        <h2>Job Details: ${job.id}</h2>
        <div class="job-header">
            <div class="job-state state-${job.state.toLowerCase()}">${job.state}</div>
        </div>

        <div class="job-stats">
            <div class="stat">
                <div class="stat-value">${job.task_count}</div>
                <div class="stat-label">Total Tasks</div>
            </div>
            <div class="stat">
                <div class="stat-value">${job.completed_tasks}</div>
                <div class="stat-label">Completed</div>
            </div>
            <div class="stat">
                <div class="stat-value">${job.running_tasks}</div>
                <div class="stat-label">Running</div>
            </div>
            <div class="stat">
                <div class="stat-value">${job.pending_tasks}</div>
                <div class="stat-label">Pending</div>
            </div>
            <div class="stat">
                <div class="stat-value">${job.failed_tasks}</div>
                <div class="stat-label">Failed</div>
            </div>
        </div>

        ${job.error ? `<div class="job-error">Error: ${job.error}</div>` : ''}

        <div class="view-toggle">
            <button class="active" onclick="switchView('list', event)">Details</button>
            <button onclick="switchView('gantt', event)">Timeline</button>
        </div>

        <div id="task-list-view">
            <div class="task-list">
                <h3>Tasks (${tasks.length})</h3>
                ${tasks.length === 0 ? '<p>No tasks found for this job.</p>' : tasks.map(task => `
                    <div class="task-item ${task.state.toLowerCase()}">
                        <div class="task-header">
                            <div class="task-id">${task.task_id}</div>
                            <div class="task-state state-${task.state.toLowerCase()}">${task.state}</div>
                        </div>

                        <div class="task-progress">
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: ${task.progress * 100}%"></div>
                            </div>
                            <div style="text-align: center; font-size: 12px; margin-top: 4px;">
                                ${Math.round(task.progress * 100)}%
                            </div>
                        </div>

                        <div class="task-details">
                            <div>Retries: ${task.retries}/${task.max_retries}</div>
                            <div>Timeout: ${task.timeout_secs}s</div>
                            <div>Waiting on: ${task.waiting_on} prerequisites</div>
                            ${task.error ? `<div style="color: #dc2626; margin-top: 5px;">Error: ${task.error}</div>` : ''}
                            ${task.started_at ? `<div>Started: ${new Date(task.started_at).toLocaleString()}</div>` : ''}
                            ${task.updated_at ? `<div>Updated: ${new Date(task.updated_at).toLocaleString()}</div>` : ''}
                        </div>
                    </div>
                `).join('')}
            </div>
        </div>

        <div id="timeline-view" class="hidden">
            <div class="full-timeline-container">
                <div class="timeline-header">
                    <h3>Task Timeline Progression</h3>
                    <div class="timeline-bounds">
                        <span class="timeline-start"></span>
                        <span class="timeline-end"></span>
                    </div>
                </div>
                <div class="timeline-container">
                    <!-- Timeline content will be dynamically generated -->
                </div>
                <div class="timeline-summary">
                    <h3>Task Summary</h3>
                    <div class="summary-stats">
                        <div class="summary-stat">
                            <div class="stat-number">${tasks.filter(t => t.state === 'done').length}</div>
                            <div class="stat-label">Completed</div>
                        </div>
                        <div class="summary-stat">
                            <div class="stat-number">${tasks.filter(t => t.state === 'running').length}</div>
                            <div class="stat-label">Running</div>
                        </div>
                        <div class="summary-stat">
                            <div class="stat-number">${tasks.filter(t => t.state === 'pending').length}</div>
                            <div class="stat-label">Pending</div>
                        </div>
                        <div class="summary-stat">
                            <div class="stat-number">${tasks.filter(t => t.state === 'failed').length}</div>
                            <div class="stat-label">Failed</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
}

function switchView(view, event) {
    console.log('switchView called with:', view, event);

    const listView = document.getElementById('task-list-view');
    const timelineView = document.getElementById('timeline-view');
    const buttons = document.querySelectorAll('.view-toggle button');

    console.log('Found elements:', { listView, timelineView, buttons: buttons.length });

    // Update button states with visual feedback
    buttons.forEach(btn => {
        btn.classList.remove('active');
        btn.style.transform = 'scale(0.95)';
    });

    // Add a small delay for visual feedback
    setTimeout(() => {
        if (event && event.target) {
            event.target.classList.add('active');
            event.target.style.transform = 'scale(1)';
        }

        if (view === 'list') {
            console.log('Switching to list view');
            listView.classList.remove('hidden');
            timelineView.classList.add('hidden');
            if (currentChart) {
                currentChart.destroy();
                currentChart = null;
            }
            // Add a subtle animation
            listView.style.opacity = '0';
            setTimeout(() => {
                listView.style.opacity = '1';
                listView.style.transition = 'opacity 0.3s ease';
            }, 50);
        } else if (view === 'gantt') {
            console.log('Switching to timeline view');
            listView.classList.add('hidden');
            timelineView.classList.remove('hidden');
            createTimelineChart();
            // Add a subtle animation
            timelineView.style.opacity = '0';
            setTimeout(() => {
                timelineView.style.opacity = '1';
                timelineView.style.transition = 'opacity 0.3s ease';
            }, 50);
        }
    }, 100);
}

function createTimelineChart() {
    if (!currentJobDetail || !currentJobDetail.tasks || currentJobDetail.tasks.length === 0) {
        return;
    }

    const tasks = currentJobDetail.tasks;

    // Create timeline data for proper Gantt chart
    const now = new Date();
    const taskData = tasks.map((task, index) => {
        const startTime = task.started_at ? new Date(task.started_at) : new Date(task.created_at);
        const endTime = task.updated_at ? new Date(task.updated_at) : now;

        return {
            task: task.task_id,
            state: task.state,
            progress: task.progress,
            startTime: startTime,
            endTime: endTime,
            error: task.error,
            retries: task.retries,
            maxRetries: task.max_retries
        };
    });

    // Sort tasks by start time for timeline progression
    taskData.sort((a, b) => a.startTime - b.startTime);

    // Find the overall timeline bounds
    const allStartTimes = taskData.map(d => d.startTime);
    const allEndTimes = taskData.map(d => d.endTime);
    const timelineStart = new Date(Math.min(...allStartTimes));
    const timelineEnd = new Date(Math.max(...allEndTimes));

    // Create timeline data for each task
    const timelineData = taskData.map((task, index) => {
        const startOffset = (task.startTime - timelineStart) / (timelineEnd - timelineStart) * 100;
        const duration = (task.endTime - task.startTime) / (timelineEnd - timelineStart) * 100;

        return {
            task: task.task,
            state: task.state,
            progress: task.progress,
            startOffset: startOffset,
            duration: duration,
            startTime: task.startTime,
            endTime: task.endTime,
            error: task.error,
            retries: task.retries,
            maxRetries: task.maxRetries
        };
    });

    // Update the timeline bounds display
    const timelineStartSpan = document.querySelector('.timeline-start');
    const timelineEndSpan = document.querySelector('.timeline-end');
    if (timelineStartSpan) timelineStartSpan.textContent = timelineStart.toLocaleString();
    if (timelineEndSpan) timelineEndSpan.textContent = timelineEnd.toLocaleString();

    // Populate the timeline container
    const timelineContainer = document.querySelector('.timeline-container');
    if (timelineContainer) {
        timelineContainer.innerHTML = `
            ${timelineData.map((task, index) => `
                <div class="timeline-row">
                    <div class="task-label">${task.task}</div>
                    <div class="timeline-bar-container">
                        <div class="timeline-bar ${task.state.toLowerCase()}"
                             style="left: ${task.startOffset}%; width: ${Math.max(task.duration, 2)}%;">
                            <div class="bar-progress" style="width: ${task.progress * 100}%"></div>
                            <div class="bar-tooltip">
                                <strong>${task.task}</strong><br>
                                State: ${task.state}<br>
                                Progress: ${Math.round(task.progress * 100)}%<br>
                                Started: ${task.startTime.toLocaleTimeString()}<br>
                                Duration: ${Math.round((task.endTime - task.startTime) / 1000)}s<br>
                                Retries: ${task.retries}/${task.maxRetries}<br>
                                ${task.error ? `Error: ${task.error}` : ''}
                            </div>
                        </div>
                    </div>
                </div>
            `).join('')}
        `;

        // Add event listeners for tooltips
        const bars = timelineContainer.querySelectorAll('.timeline-bar');
        bars.forEach(bar => {
            bar.addEventListener('mouseenter', function() {
                this.querySelector('.bar-tooltip').style.display = 'block';
            });
            bar.addEventListener('mouseleave', function() {
                this.querySelector('.bar-tooltip').style.display = 'none';
            });
        });
    }
}

function closeModal() {
    document.getElementById('jobModal').style.display = 'none';
    if (currentChart) {
        currentChart.destroy();
        currentChart = null;
    }
    currentJobDetail = null;
}

// Close modal when clicking outside
window.onclick = function(event) {
    const modal = document.getElementById('jobModal');
    if (event.target === modal) {
        closeModal();
    }
}

// Close modal with Escape key
document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        closeModal();
    }
});

// Auto-refresh every 10 seconds (increased from 5 for better performance)
setInterval(loadJobs, 10000);

// Initial load
document.addEventListener('DOMContentLoaded', function() {
    loadJobs();
});

