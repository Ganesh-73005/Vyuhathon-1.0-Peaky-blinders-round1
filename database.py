"""
Database Layer for Multi-Agent Application
SQLite-based persistence with connection pooling
"""
import sqlite3
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from contextlib import contextmanager
from threading import Lock

logger = logging.getLogger(__name__)

# Database file path
DB_PATH = "database/app.db"
DB_VERSION = 1

# Thread-safe connection pool
_connection_lock = Lock()

@contextmanager
def get_db_connection():
    """Context manager for database connections with automatic commit/rollback."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row  # Enable column access by name
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        conn.close()

def initialize_database():
    """Create all database tables if they don't exist."""
    import os
    os.makedirs("database", exist_ok=True)
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Users table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            preferences TEXT DEFAULT '{}'
        )
        """)
        
        # Sessions table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            title TEXT DEFAULT 'New Chat',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            context_summary TEXT,
            is_active BOOLEAN DEFAULT 1,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """)
        
        # Messages table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
            content TEXT NOT NULL,
            formatted_content TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
        )
        """)
        
        # Generated content table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS generated_content (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            type TEXT NOT NULL CHECK(type IN ('blog', 'post', 'image')),
            content TEXT NOT NULL,
            status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'approved', 'rejected')),
            preview_url TEXT,
            final_url TEXT,
            metadata TEXT DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
        )
        """)

        # Workflow variant tracking table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS workflow_variants (
            option_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            content_id TEXT NOT NULL,
            workflow_name TEXT NOT NULL,
            state_hash TEXT NOT NULL,
            label TEXT,
            metadata TEXT DEFAULT '{}',
            is_selected INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE,
            FOREIGN KEY (content_id) REFERENCES generated_content(id) ON DELETE CASCADE
        )
        """)
        
        # Social posts table (for tracking published posts)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS social_posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content_id TEXT NOT NULL,
            platform TEXT NOT NULL CHECK(platform IN ('twitter', 'instagram', 'facebook', 'linkedin')),
            post_id TEXT,
            post_url TEXT,
            posted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (content_id) REFERENCES generated_content(id) ON DELETE CASCADE
        )
        """)
        
        # Agent costs table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS agent_costs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_name TEXT UNIQUE NOT NULL,
            token_cost REAL DEFAULT 0,
            time_cost REAL DEFAULT 0,
            api_cost_per_call REAL DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Execution metrics table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS execution_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            agent_name TEXT NOT NULL,
            execution_time REAL NOT NULL,
            tokens_used INTEGER DEFAULT 0,
            cost REAL DEFAULT 0,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
        )
        """)
        
        # Social metrics table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS social_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content_id TEXT NOT NULL,
            platform TEXT NOT NULL CHECK(platform IN ('twitter', 'instagram', 'facebook', 'linkedin')),
            likes INTEGER DEFAULT 0,
            comments INTEGER DEFAULT 0,
            shares INTEGER DEFAULT 0,
            impressions INTEGER DEFAULT 0,
            reach INTEGER DEFAULT 0,
            engagement_rate REAL DEFAULT 0,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (content_id) REFERENCES generated_content(id) ON DELETE CASCADE
        )
        """)
        
        # RL state table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS rl_state (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            state_vector TEXT NOT NULL,
            action_taken TEXT NOT NULL,
            reward REAL NOT NULL,
            next_state_vector TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Q-table for Q-learning (deprecated - kept for backward compatibility)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS q_table (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            state_hash TEXT NOT NULL,
            action TEXT NOT NULL,
            q_value REAL DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(state_hash, action)
        )
        """)
        
        # MABO Coordination State table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS mabo_coordination_state (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            iteration INTEGER NOT NULL,
            coordination_vars TEXT NOT NULL,
            lagrange_multipliers TEXT NOT NULL,
            budget_allocations TEXT NOT NULL,
            total_budget REAL NOT NULL,
            convergence_threshold REAL DEFAULT 0.0001,
            last_update TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(iteration)
        )
        """)
        
        # MABO Local BO State table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS mabo_local_bo_state (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_name TEXT NOT NULL,
            observed_points TEXT NOT NULL,
            observed_values TEXT NOT NULL,
            best_point TEXT,
            best_value REAL DEFAULT 999999.0,
            iteration INTEGER DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(agent_name)
        )
        """)
        
        # MABO Reward Queue table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS mabo_reward_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content_id TEXT NOT NULL,
            state_hash TEXT NOT NULL,
            action TEXT NOT NULL,
            expected_delay_hours REAL NOT NULL,
            reward REAL,
            engagement_rate REAL,
            cost REAL,
            execution_time REAL,
            content_approved INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            stabilized_at TIMESTAMP,
            is_stabilized INTEGER DEFAULT 0,
            UNIQUE(content_id)
        )
        """)
        
        # Budget Allocator Campaign State table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS budget_allocator_campaigns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            campaign_id TEXT NOT NULL,
            total_budget_spent REAL DEFAULT 0.0,
            total_clicks INTEGER DEFAULT 0,
            total_impressions INTEGER DEFAULT 0,
            observations TEXT,
            is_censored INTEGER DEFAULT 0,
            parameters TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(campaign_id)
        )
        """)
        
        # Validation Metrics table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS validation_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metric_type TEXT NOT NULL,
            value REAL NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create indexes for MABO tables
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_mabo_coordination_iter ON mabo_coordination_state(iteration)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_mabo_local_bo_agent ON mabo_local_bo_state(agent_name)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_mabo_reward_content ON mabo_reward_queue(content_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_mabo_reward_stabilized ON mabo_reward_queue(is_stabilized)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_budget_campaign ON budget_allocator_campaigns(campaign_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_validation_metrics_type ON validation_metrics(metric_type, timestamp)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_workflow_variants_state ON workflow_variants(state_hash)")
        
        # Brand profiles table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS brand_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            brand_name TEXT NOT NULL,
            contacts TEXT,
            location TEXT,
            logo_url TEXT,
            metadata TEXT DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )
        """)
        
        # Create indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_session ON generated_content(session_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_metrics_content ON social_metrics(content_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_q_table_lookup ON q_table(state_hash, action)")
        
        # Initialize default agent costs
        default_costs = [
            ('webcrawler', 0.0001, 10, 0),
            ('seo_agent', 0.0001, 30, 0),
            ('keyword_extractor', 0.0005, 15, 0),
            ('gap_analyzer', 0.001, 20, 0.005),
            ('content_agent_blog', 0.002, 25, 0),
            ('content_agent_social', 0.0005, 10, 0),
            ('image_generator', 0, 45, 0.05),
            ('social_poster', 0, 5, 0)
        ]
        
        for agent_name, token_cost, time_cost, api_cost in default_costs:
            cursor.execute("""
            INSERT OR IGNORE INTO agent_costs (agent_name, token_cost, time_cost, api_cost_per_call)
            VALUES (?, ?, ?, ?)
            """, (agent_name, token_cost, time_cost, api_cost))
        
        logger.info(f"Database initialized successfully at {DB_PATH}")

# ==================== USER OPERATIONS ====================

def create_user(email: str, password_hash: str) -> int:
    """Create a new user and return user_id."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO users (email, password_hash, created_at)
        VALUES (?, ?, ?)
        """, (email, password_hash, datetime.now()))
        return cursor.lastrowid

def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """Get user by email address."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = ?", (email,))
        row = cursor.fetchone()
        return dict(row) if row else None

def get_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """Get user by ID."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

def update_last_login(user_id: int):
    """Update user's last login timestamp."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        UPDATE users SET last_login = ? WHERE id = ?
        """, (datetime.now(), user_id))

# ==================== SESSION OPERATIONS ====================

def create_session(session_id: str, user_id: int, title: str = "New Chat") -> str:
    """Create a new session for a user."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO sessions (id, user_id, title, created_at, last_active)
        VALUES (?, ?, ?, ?, ?)
        """, (session_id, user_id, title, datetime.now(), datetime.now()))
        return session_id

def get_session(session_id: str, user_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
    """Get session by ID, optionally verifying user ownership."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if user_id:
            cursor.execute("""
            SELECT * FROM sessions WHERE id = ? AND user_id = ? AND is_active = 1
            """, (session_id, user_id))
        else:
            cursor.execute("SELECT * FROM sessions WHERE id = ? AND is_active = 1", (session_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

def get_user_sessions(user_id: int, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    """Get all sessions for a user, ordered by last_active."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        SELECT * FROM sessions 
        WHERE user_id = ? AND is_active = 1 
        ORDER BY last_active DESC 
        LIMIT ? OFFSET ?
        """, (user_id, limit, offset))
        return [dict(row) for row in cursor.fetchall()]

def update_session_activity(session_id: str):
    """Update session's last_active timestamp."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        UPDATE sessions SET last_active = ? WHERE id = ?
        """, (datetime.now(), session_id))

def update_session_title(session_id: str, title: str):
    """Update session title."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        UPDATE sessions SET title = ? WHERE id = ?
        """, (title, session_id))

def update_session_context(session_id: str, context_summary: str):
    """Update session context summary."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        UPDATE sessions SET context_summary = ? WHERE id = ?
        """, (context_summary, session_id))

def delete_session(session_id: str, user_id: int):
    """Soft delete a session."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        UPDATE sessions SET is_active = 0 WHERE id = ? AND user_id = ?
        """, (session_id, user_id))

# ==================== MESSAGE OPERATIONS ====================

def save_message(session_id: str, role: str, content: str, formatted_content: Optional[str] = None):
    """Save a message to the database."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO messages (session_id, role, content, formatted_content, timestamp)
        VALUES (?, ?, ?, ?, ?)
        """, (session_id, role, content, formatted_content, datetime.now()))

def get_session_messages(session_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Get all messages for a session."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if limit:
            cursor.execute("""
            SELECT * FROM messages 
            WHERE session_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
            """, (session_id, limit))
        else:
            cursor.execute("""
            SELECT * FROM messages 
            WHERE session_id = ? 
            ORDER BY timestamp ASC
            """, (session_id,))
        return [dict(row) for row in cursor.fetchall()]

def get_recent_messages(session_id: str, count: int = 10) -> List[Dict[str, Any]]:
    """Get the most recent N messages for a session."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        SELECT * FROM messages 
        WHERE session_id = ? 
        ORDER BY timestamp DESC 
        LIMIT ?
        """, (session_id, count))
        messages = [dict(row) for row in cursor.fetchall()]
        return list(reversed(messages))  # Return in chronological order

# ==================== CONTENT OPERATIONS ====================

def save_generated_content(content_id: str, session_id: str, content_type: str, 
                          content: str, preview_url: Optional[str] = None,
                          metadata: Optional[Dict] = None) -> str:
    """Save generated content (blog/post)."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        metadata_json = json.dumps(metadata) if metadata else '{}'
        cursor.execute("""
        INSERT INTO generated_content (id, session_id, type, content, preview_url, metadata, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (content_id, session_id, content_type, content, preview_url, metadata_json, datetime.now()))
        return content_id

def update_content_metadata(content_id: str, metadata_updates: Dict[str, Any]):
    """Merge updates into generated content metadata."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT metadata FROM generated_content WHERE id = ?", (content_id,))
        row = cursor.fetchone()
        if not row:
            return
        current_metadata = json.loads(row[0]) if row[0] else {}
        current_metadata.update(metadata_updates)
        cursor.execute("""
        UPDATE generated_content SET metadata = ? WHERE id = ?
        """, (json.dumps(current_metadata), content_id))

def save_workflow_variant(option_id: str, session_id: str, content_id: str,
                          workflow_name: str, state_hash: str,
                          label: Optional[str] = None,
                          metadata: Optional[Dict[str, Any]] = None):
    """Persist workflow variant metadata."""
    metadata_json = json.dumps(metadata or {})
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        INSERT OR REPLACE INTO workflow_variants
        (option_id, session_id, content_id, workflow_name, state_hash, label, metadata, is_selected, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, 
                COALESCE((SELECT is_selected FROM workflow_variants WHERE option_id = ?), 0),
                COALESCE((SELECT created_at FROM workflow_variants WHERE option_id = ?), ?))
        """, (
            option_id,
            session_id,
            content_id,
            workflow_name,
            state_hash,
            label,
            metadata_json,
            option_id,
            option_id,
            datetime.now()
        ))

def get_workflow_variant(option_id: str) -> Optional[Dict[str, Any]]:
    """Fetch workflow variant by option id."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM workflow_variants WHERE option_id = ?", (option_id,))
        row = cursor.fetchone()
        if row:
            data = dict(row)
            data['metadata'] = json.loads(data['metadata']) if data['metadata'] else {}
            return data
        return None

def get_workflow_variants(session_id: str, state_hash: str) -> List[Dict[str, Any]]:
    """Get all variants for a given session/state hash."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        SELECT * FROM workflow_variants
        WHERE session_id = ? AND state_hash = ?
        ORDER BY created_at ASC
        """, (session_id, state_hash))
        variants = []
        for row in cursor.fetchall():
            data = dict(row)
            data['metadata'] = json.loads(data['metadata']) if data['metadata'] else {}
            variants.append(data)
        return variants

def mark_variant_selection(option_id: str, is_selected: bool):
    """Mark a workflow variant as selected or not."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        UPDATE workflow_variants SET is_selected = ? WHERE option_id = ?
        """, (1 if is_selected else 0, option_id))

def clear_variant_selection(session_id: str, state_hash: str, exclude_option_id: Optional[str] = None):
    """Clear selection flags for all variants in a group, optionally excluding one."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if exclude_option_id:
            cursor.execute("""
            UPDATE workflow_variants
            SET is_selected = 0
            WHERE session_id = ? AND state_hash = ? AND option_id != ?
            """, (session_id, state_hash, exclude_option_id))
        else:
            cursor.execute("""
            UPDATE workflow_variants
            SET is_selected = 0
            WHERE session_id = ? AND state_hash = ?
            """, (session_id, state_hash))
def get_generated_content(content_id: str) -> Optional[Dict[str, Any]]:
    """Get generated content by ID."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM generated_content WHERE id = ?", (content_id,))
        row = cursor.fetchone()
        if row:
            content = dict(row)
            content['metadata'] = json.loads(content['metadata'])
            return content
        return None

def update_content_status(content_id: str, status: str, final_url: Optional[str] = None):
    """Update content status (pending/approved/rejected)."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        if final_url:
            cursor.execute("""
            UPDATE generated_content SET status = ?, final_url = ? WHERE id = ?
            """, (status, final_url, content_id))
        else:
            cursor.execute("""
            UPDATE generated_content SET status = ? WHERE id = ?
            """, (status, content_id))

# ==================== METRICS OPERATIONS ====================

def log_execution_metrics(session_id: str, agent_name: str, execution_time: float,
                          tokens_used: int = 0, cost: float = 0):
    """Log agent execution metrics."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO execution_metrics (session_id, agent_name, execution_time, tokens_used, cost, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (session_id, agent_name, execution_time, tokens_used, cost, datetime.now()))

def save_social_metrics(content_id: str, platform: str, likes: int = 0, comments: int = 0,
                       shares: int = 0, impressions: int = 0, reach: int = 0):
    """Save social media metrics for content."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        engagement_rate = (likes + comments + shares) / max(impressions, 1) * 100
        cursor.execute("""
        INSERT INTO social_metrics 
        (content_id, platform, likes, comments, shares, impressions, reach, engagement_rate, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (content_id, platform, likes, comments, shares, impressions, reach, engagement_rate, datetime.now()))

def get_social_metrics(content_id: Optional[str] = None, user_id: Optional[int] = None,
                       platform: Optional[str] = None, days: int = 30) -> List[Dict[str, Any]]:
    """Get social metrics with optional filters."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        query = """
        SELECT sm.* FROM social_metrics sm
        JOIN generated_content gc ON sm.content_id = gc.id
        JOIN sessions s ON gc.session_id = s.id
        WHERE sm.timestamp >= datetime('now', '-' || ? || ' days')
        """
        params = [days]
        
        if content_id:
            query += " AND sm.content_id = ?"
            params.append(content_id)
        if user_id:
            query += " AND s.user_id = ?"
            params.append(user_id)
        if platform:
            query += " AND sm.platform = ?"
            params.append(platform)
        
        query += " ORDER BY sm.timestamp DESC"
        cursor.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

# ==================== BRAND PROFILE OPERATIONS ====================

def save_brand_profile(user_id: int, brand_name: str, contacts: Optional[str] = None,
                      location: Optional[str] = None, logo_url: Optional[str] = None,
                      metadata: Optional[Dict] = None) -> int:
    """Save or update brand profile for user."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        metadata_json = json.dumps(metadata) if metadata else '{}'
        
        # Check if profile exists
        cursor.execute("SELECT id FROM brand_profiles WHERE user_id = ?", (user_id,))
        existing = cursor.fetchone()
        
        if existing:
            cursor.execute("""
            UPDATE brand_profiles 
            SET brand_name = ?, contacts = ?, location = ?, logo_url = ?, metadata = ?, updated_at = ?
            WHERE user_id = ?
            """, (brand_name, contacts, location, logo_url, metadata_json, datetime.now(), user_id))
            return existing[0]
        else:
            cursor.execute("""
            INSERT INTO brand_profiles (user_id, brand_name, contacts, location, logo_url, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (user_id, brand_name, contacts, location, logo_url, metadata_json, datetime.now()))
            return cursor.lastrowid

def get_brand_profile(user_id: int) -> Optional[Dict[str, Any]]:
    """Get brand profile for user."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM brand_profiles WHERE user_id = ? ORDER BY updated_at DESC LIMIT 1", (user_id,))
        row = cursor.fetchone()
        if row:
            profile = dict(row)
            profile['metadata'] = json.loads(profile['metadata'])
            return profile
        return None

def update_brand_profile(user_id: int, brand_name: Optional[str] = None, contacts: Optional[str] = None,
                       location: Optional[str] = None, logo_url: Optional[str] = None,
                       metadata: Optional[Dict] = None) -> bool:
    """Update brand profile for user (partial update)."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Get existing profile
        cursor.execute("SELECT * FROM brand_profiles WHERE user_id = ? ORDER BY updated_at DESC LIMIT 1", (user_id,))
        existing = cursor.fetchone()
        
        if not existing:
            # Create new profile if doesn't exist
            if brand_name:
                metadata_json = json.dumps(metadata) if metadata else '{}'
                cursor.execute("""
                INSERT INTO brand_profiles (user_id, brand_name, contacts, location, logo_url, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (user_id, brand_name, contacts, location, logo_url, metadata_json, datetime.now()))
                return True
            return False
        
        # Build update query with only provided fields
        updates = []
        values = []
        
        if brand_name is not None:
            updates.append("brand_name = ?")
            values.append(brand_name)
        if contacts is not None:
            updates.append("contacts = ?")
            values.append(contacts)
        if location is not None:
            updates.append("location = ?")
            values.append(location)
        if logo_url is not None:
            updates.append("logo_url = ?")
            values.append(logo_url)
        if metadata is not None:
            # Merge with existing metadata
            existing_metadata = json.loads(existing['metadata']) if existing['metadata'] else {}
            if isinstance(metadata, dict):
                existing_metadata.update(metadata)
            updates.append("metadata = ?")
            values.append(json.dumps(existing_metadata))
        
        if not updates:
            return False
        
        updates.append("updated_at = ?")
        values.append(datetime.now())
        values.append(user_id)
        
        query = f"UPDATE brand_profiles SET {', '.join(updates)} WHERE user_id = ?"
        cursor.execute(query, values)
        return True

def delete_brand_profile(user_id: int):
    """Delete brand profile for user."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM brand_profiles WHERE user_id = ?", (user_id,))
        conn.commit()
        logger.info(f"Deleted brand profile for user {user_id}")

def save_social_post(content_id: str, platform: str, post_url: str):
    """Save social media post for metrics tracking."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO social_posts (content_id, platform, post_id, post_url, posted_at)
        VALUES (?, ?, ?, ?, ?)
        """, (content_id, platform, content_id, post_url, datetime.now()))
        conn.commit()
        logger.info(f"Social post saved: {platform} - {post_url}")

# ==================== RL OPERATIONS ====================

def save_rl_experience(state_vector: str, action_taken: str, reward: float, next_state_vector: Optional[str] = None):
    """Save RL experience for training."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO rl_state (state_vector, action_taken, reward, next_state_vector, timestamp)
        VALUES (?, ?, ?, ?, ?)
        """, (state_vector, action_taken, reward, next_state_vector, datetime.now()))

def get_q_value(state_hash: str, action: str) -> float:
    """Get Q-value for state-action pair."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT q_value FROM q_table WHERE state_hash = ? AND action = ?", (state_hash, action))
        row = cursor.fetchone()
        return row[0] if row else 0.0

def update_q_value(state_hash: str, action: str, q_value: float):
    """Update Q-value for state-action pair."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO q_table (state_hash, action, q_value, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(state_hash, action) DO UPDATE SET
        q_value = excluded.q_value,
        updated_at = excluded.updated_at
        """, (state_hash, action, q_value, datetime.now()))

def get_all_q_values(state_hash: str) -> Dict[str, float]:
    """Get all Q-values for a given state."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT action, q_value FROM q_table WHERE state_hash = ?", (state_hash,))
        return {row[0]: row[1] for row in cursor.fetchall()}

# Initialize database on module import
try:
    initialize_database()
except Exception as e:
    logger.error(f"Failed to initialize database: {e}")

