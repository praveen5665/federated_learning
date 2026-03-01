# -*- coding: utf-8 -*-
"""
Federated Learning Dashboard - GUI for Server and Client Management
Provides a unified interface to start/stop server and clients
"""

import sys
import io
# Force UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import re
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import queue
import subprocess
import threading
import os
import json
from datetime import datetime

class FederatedLearningDashboard:
    """Main GUI Dashboard for Federated Learning"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Federated Learning Dashboard - IoT Anomaly Detection")
        self.root.geometry("1400x900")
        self.root.configure(bg='#F3F6FB')

        # Visual theme
        self.colors = {
            "bg": "#F3F6FB",
            "surface": "#FFFFFF",
            "surface_alt": "#EEF3FB",
            "border": "#D7E1F0",
            "text": "#1F2A3A",
            "text_muted": "#6D7B91",
            "success": "#1FA97A",
            "danger": "#E45858",
            "warning": "#E9A23B",
            "info": "#3D7BEB",
            "purple": "#7567F8",
            "teal": "#1AA6A6",
            "console": "#F8FAFD",
            "console_server": "#197E58",
            "console_client": "#1D5FCF"
        }
        
        # Process tracking
        self.server_process = None
        self.client_processes = {}  # {client_id: process}
        self.max_clients = 10  
        
        # Output queues for threading
        self.server_queue = queue.Queue()
        self.client_queues = {i: queue.Queue() for i in range(1, self.max_clients + 1)}
        
        # Status tracking
        self.server_status = "Stopped"
        self.client_statuses = {i: "Stopped" for i in range(1, self.max_clients + 1)}
        
        self.async_stats = {
            "fedbuff": "N/A",
            "version": "0",
            "buffered": "0",
            "applied": "0",
            "avg_staleness": "0.0000",
            "status": "Unknown",
        }
        self.async_labels = {}
        
        self.setup_theme()
        self.setup_ui()
        self.check_prerequisites()
        
        # Start queue monitoring
        self.root.after(100, self.process_queues)

    def setup_theme(self):
        """Configure ttk theme and shared styles"""
        self.title_font = ("Segoe UI", 22, "bold")
        self.section_font = ("Segoe UI", 12, "bold")
        self.body_font = ("Segoe UI", 10)
        self.code_font = ("Cascadia Code", 9)

        style = ttk.Style()
        style.theme_use('clam')

        style.configure(
            'TNotebook',
            background=self.colors["bg"],
            borderwidth=0,
            tabmargins=[8, 8, 8, 0]
        )
        style.configure(
            'TNotebook.Tab',
            background=self.colors["surface_alt"],
            foreground=self.colors["text_muted"],
            padding=[20, 10],
            font=('Segoe UI', 10, 'bold'),
            borderwidth=1
        )
        style.map(
            'TNotebook.Tab',
            background=[('selected', self.colors["surface"]), ('active', self.colors["surface"])],
            foreground=[('selected', self.colors["text"]), ('active', self.colors["text"])],
        )

        style.configure('Dark.TFrame', background=self.colors["surface"])
        style.configure('Dark.TLabel', background=self.colors["surface"], foreground=self.colors["text"])

    def create_primary_button(self, parent, text, command, color):
        """Create consistently styled action button"""
        return tk.Button(
            parent,
            text=text,
            font=("Segoe UI Semibold", 10),
            bg=color,
            fg='white',
            activebackground=color,
            activeforeground='white',
            relief=tk.FLAT,
            bd=0,
            padx=18,
            pady=9,
            command=command,
            cursor='hand2',
            highlightthickness=1,
            highlightbackground=color,
            disabledforeground="#B8C2D3"
        )
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main container with padding
        main_container = tk.Frame(self.root, bg=self.colors["bg"])
        main_container.pack(fill=tk.BOTH, expand=True, padx=18, pady=18)
        
        # Title
        title_frame = tk.Frame(
            main_container,
            bg=self.colors["surface"],
            relief=tk.FLAT,
            bd=1,
            highlightthickness=1,
            highlightbackground=self.colors["border"]
        )
        title_frame.pack(fill=tk.X, pady=(0, 14))
        
        title = tk.Label(
            title_frame,
            text="Federated Learning Control Center",
            font=self.title_font,
            bg=self.colors["surface"],
            fg=self.colors["text"],
            pady=8
        )
        title.pack()

        subtitle = tk.Label(
            title_frame,
            text="Model orchestration for server, clients, logs, and experiment results",
            font=("Segoe UI", 10),
            bg=self.colors["surface"],
            fg=self.colors["text_muted"],
            pady=0
        )
        subtitle.pack(pady=(0, 12))
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.setup_overview_tab()
        self.setup_server_tab()
        self.setup_clients_tab()
        self.setup_logs_tab()
        self.setup_results_tab()
    
    def setup_overview_tab(self):
        """Overview tab with system status"""
        overview_frame = tk.Frame(self.notebook, bg=self.colors["surface"])
        self.notebook.add(overview_frame, text="Overview")
        
        # Status panel
        status_panel = tk.LabelFrame(
            overview_frame,
            text="System Status",
            font=self.section_font,
            bg=self.colors["surface"],
            fg=self.colors["text"],
            bd=1,
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground=self.colors["border"],
            padx=20,
            pady=20
        )
        status_panel.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
        
        # Server status
        server_frame = tk.Frame(status_panel, bg=self.colors["surface"])
        server_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(
            server_frame,
            text="Server:",
            font=("Segoe UI", 11, "bold"),
            bg=self.colors["surface"],
            fg=self.colors["text"],
            width=15,
            anchor='w'
        ).pack(side=tk.LEFT, padx=5)
        
        self.server_status_label = tk.Label(
            server_frame,
            text="● Stopped",
            font=("Segoe UI", 11),
            bg=self.colors["surface"],
            fg=self.colors["danger"],
            width=20,
            anchor='w'
        )
        self.server_status_label.pack(side=tk.LEFT, padx=5)
        
        # Client statuses
        self.client_status_labels = {}
        for i in range(1, self.max_clients + 1):
            client_frame = tk.Frame(status_panel, bg=self.colors["surface"])
            client_frame.pack(fill=tk.X, pady=5)
            
            tk.Label(
                client_frame,
                text=f"Client {i}:",
                font=("Segoe UI", 11),
                bg=self.colors["surface"],
                fg=self.colors["text"],
                width=15,
                anchor='w'
            ).pack(side=tk.LEFT, padx=5)
            
            status_label = tk.Label(
                client_frame,
                text="● Stopped",
                font=("Segoe UI", 11),
                bg=self.colors["surface"],
                fg=self.colors["text_muted"],
                width=20,
                anchor='w'
            )
            status_label.pack(side=tk.LEFT, padx=5)
            self.client_status_labels[i] = status_label
        
        # Quick actions
        actions_frame = tk.LabelFrame(
            overview_frame,
            text="Quick Actions",
            font=self.section_font,
            bg=self.colors["surface"],
            fg=self.colors["text"],
            bd=1,
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground=self.colors["border"],
            padx=20,
            pady=20
        )
        actions_frame.pack(fill=tk.X, padx=12, pady=(0, 12))
        
        btn_frame = tk.Frame(actions_frame, bg=self.colors["surface"])
        btn_frame.pack()
        
        self.create_primary_button(btn_frame, "Start Full System", self.start_full_system, self.colors["success"]).pack(side=tk.LEFT, padx=10)
        self.create_primary_button(btn_frame, "Stop All", self.stop_all, self.colors["danger"]).pack(side=tk.LEFT, padx=10)
        self.create_primary_button(btn_frame, "Generate Results", self.generate_results, self.colors["info"]).pack(side=tk.LEFT, padx=10)
    
    def setup_server_tab(self):
        """Server control tab"""
        server_frame = tk.Frame(self.notebook, bg=self.colors["surface"])
        self.notebook.add(server_frame, text="Server")
        
        # Control panel
        control_panel = tk.LabelFrame(
            server_frame,
            text="Server Control",
            font=self.section_font,
            bg=self.colors["surface"],
            fg=self.colors["text"],
            bd=1,
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground=self.colors["border"],
            padx=20,
            pady=20
        )
        control_panel.pack(fill=tk.X, padx=12, pady=12)
        
        # Rounds input
        rounds_frame = tk.Frame(control_panel, bg=self.colors["surface"])
        rounds_frame.pack(pady=10)
        
        tk.Label(
            rounds_frame,
            text="Number of Rounds:",
            font=("Segoe UI", 11),
            bg=self.colors["surface"],
            fg=self.colors["text"]
        ).pack(side=tk.LEFT, padx=5)
        
        self.rounds_var = tk.StringVar(value="10")
        rounds_entry = tk.Entry(
            rounds_frame,
            textvariable=self.rounds_var,
            font=("Segoe UI", 11),
            width=10,
            bg=self.colors["surface_alt"],
            fg=self.colors["text"],
            insertbackground=self.colors["text"],
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground=self.colors["border"],
            highlightcolor=self.colors["info"]
        )
        rounds_entry.pack(side=tk.LEFT, padx=5)
        
        # Buttons
        btn_frame = tk.Frame(control_panel, bg=self.colors["surface"])
        btn_frame.pack(pady=10)
        
        self.start_server_btn = self.create_primary_button(btn_frame, "Start Server", self.start_server, self.colors["success"])
        self.start_server_btn.pack(side=tk.LEFT, padx=10)
        
        self.stop_server_btn = self.create_primary_button(btn_frame, "Stop Server", self.stop_server, self.colors["danger"])
        self.stop_server_btn.config(state=tk.DISABLED)
        self.stop_server_btn.pack(side=tk.LEFT, padx=10)
        
        # Server output
        output_panel = tk.LabelFrame(
            server_frame,
            text="Server Output",
            font=("Segoe UI", 11, "bold"),
            bg=self.colors["surface"],
            fg=self.colors["text"],
            bd=1,
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground=self.colors["border"],
            padx=10,
            pady=10
        )
        output_panel.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))
        
        self.server_output = scrolledtext.ScrolledText(
            output_panel,
            font=self.code_font,
            bg=self.colors["console"],
            fg=self.colors["console_server"],
            insertbackground=self.colors["text"],
            wrap=tk.WORD,
            relief=tk.FLAT,
            bd=0,
            padx=8,
            pady=8
        )
        self.server_output.pack(fill=tk.BOTH, expand=True)
    
    def setup_clients_tab(self):
        """Clients control tab"""
        clients_frame = tk.Frame(self.notebook, bg=self.colors["surface"])
        self.notebook.add(clients_frame, text="Clients")
        
        # Create sub-tabs for each client
        self.client_notebook = ttk.Notebook(clients_frame)
        self.client_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.client_tabs = {}
        self.client_outputs = {}
        self.client_start_btns = {}
        self.client_stop_btns = {}
        
        for i in range(1, self.max_clients + 1):
            self.create_client_tab(i)
    
    def create_client_tab(self, client_id):
        """Create tab for individual client"""
        client_frame = tk.Frame(self.client_notebook, bg=self.colors["surface"])
        self.client_notebook.add(client_frame, text=f"Client {client_id}")
        self.client_tabs[client_id] = client_frame
        
        # Control panel
        control_panel = tk.LabelFrame(
            client_frame,
            text=f"Client {client_id} Control",
            font=("Segoe UI", 11, "bold"),
            bg=self.colors["surface"],
            fg=self.colors["text"],
            bd=1,
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground=self.colors["border"],
            padx=20,
            pady=15
        )
        control_panel.pack(fill=tk.X, padx=10, pady=10)
        
        btn_frame = tk.Frame(control_panel, bg=self.colors["surface"])
        btn_frame.pack()
        
        start_btn = self.create_primary_button(
            btn_frame,
            f"Start Client {client_id}",
            lambda: self.start_client(client_id),
            self.colors["success"]
        )
        start_btn.pack(side=tk.LEFT, padx=10)
        self.client_start_btns[client_id] = start_btn
        
        stop_btn = self.create_primary_button(
            btn_frame,
            f"Stop Client {client_id}",
            lambda: self.stop_client(client_id),
            self.colors["danger"]
        )
        stop_btn.config(state=tk.DISABLED)
        stop_btn.pack(side=tk.LEFT, padx=10)
        self.client_stop_btns[client_id] = stop_btn
        
        # Output
        output_panel = tk.LabelFrame(
            client_frame,
            text=f"Client {client_id} Output",
            font=("Segoe UI", 11, "bold"),
            bg=self.colors["surface"],
            fg=self.colors["text"],
            bd=1,
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground=self.colors["border"],
            padx=10,
            pady=10
        )
        output_panel.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        output = scrolledtext.ScrolledText(
            output_panel,
            font=self.code_font,
            bg=self.colors["console"],
            fg=self.colors["console_client"],
            insertbackground=self.colors["text"],
            wrap=tk.WORD,
            relief=tk.FLAT,
            bd=0,
            padx=8,
            pady=8
        )
        output.pack(fill=tk.BOTH, expand=True)
        self.client_outputs[client_id] = output
    
    def setup_logs_tab(self):
        """Combined logs tab"""
        logs_frame = tk.Frame(self.notebook, bg=self.colors["surface"])
        self.notebook.add(logs_frame, text="All Logs")
        
        self.all_logs = scrolledtext.ScrolledText(
            logs_frame,
            font=self.code_font,
            bg=self.colors["console"],
            fg=self.colors["text"],
            insertbackground=self.colors["text"],
            wrap=tk.WORD,
            relief=tk.FLAT,
            bd=0,
            padx=8,
            pady=8
        )
        self.all_logs.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def setup_results_tab(self):
        """Results visualization tab"""
        results_frame = tk.Frame(self.notebook, bg=self.colors["surface"])
        self.notebook.add(results_frame, text="Results")
        
        control_frame = tk.Frame(results_frame, bg=self.colors["surface"])
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.create_primary_button(control_frame, "Refresh Results", self.load_results, self.colors["info"]).pack(side=tk.LEFT, padx=5)
        self.create_primary_button(control_frame, "Generate Metrics", self.generate_results, self.colors["purple"]).pack(side=tk.LEFT, padx=5)
        self.create_primary_button(control_frame, "Open Results Folder", self.open_results_folder, self.colors["teal"]).pack(side=tk.LEFT, padx=5)
        
        # Results display
        self.results_text = scrolledtext.ScrolledText(
            results_frame,
            font=self.code_font,
            bg=self.colors["console"],
            fg=self.colors["text"],
            wrap=tk.WORD,
            relief=tk.FLAT,
            bd=0,
            padx=10,
            pady=10
        )
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def check_prerequisites(self):
        """Check if required files exist"""
        required_files = [
            "server_autoencoder.py",
            "client_autoencoder.py",
            "data/processed/test_data.csv",
            "data/processed/selected_features.csv"
        ]
        
        missing = []
        for file in required_files:
            if not os.path.exists(file):
                missing.append(file)
        
        if missing:
            self.log_all(f"[WARNING] Missing files:\n" + "\n".join(f"  - {f}" for f in missing), "warning")
        else:
            self.log_all("[OK] All prerequisites found!", "success")
    
    def start_server(self):
        """Start the federated learning server"""
        if self.server_process is not None:
            messagebox.showwarning("Warning", "Server is already running!")
            return
        
        try:
            rounds = int(self.rounds_var.get())
            if rounds < 1:
                raise ValueError()
        except:
            messagebox.showerror("Error", "Please enter a valid number of rounds (>= 1)")
            return
        
        self.log_server("[START] Federated Learning Server...")
        self.log_server(f"   Rounds: {rounds}")
        self.log_server(f"   Waiting for clients to connect...\n")
        
        # Start server process
        cmd = [sys.executable, "server_autoencoder.py", str(rounds)]
        self.server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Start thread to read output
        threading.Thread(
            target=self.read_process_output,
            args=(self.server_process, self.server_queue, "SERVER"),
            daemon=True
        ).start()
        
        # Update UI
        self.server_status = "Running"
        self.update_server_status()
        self.start_server_btn.config(state=tk.DISABLED)
        self.stop_server_btn.config(state=tk.NORMAL)
        
        self.log_all("[OK] Server started successfully", "success")
    
    def stop_server(self):
        """Stop the server"""
        if self.server_process is None:
            return
        
        self.log_server("[STOP] Stopping server...")
        try:
            self.server_process.terminate()
            self.server_process.wait(timeout=5)
        except:
            self.server_process.kill()
        
        self.server_process = None
        self.server_status = "Stopped"
        self.update_server_status()
        
        self.start_server_btn.config(state=tk.NORMAL)
        self.stop_server_btn.config(state=tk.DISABLED)
        
        self.log_all("[OK] Server stopped", "info")
    
    def start_client(self, client_id):
        """Start a specific client"""
        if client_id in self.client_processes:
            messagebox.showwarning("Warning", f"Client {client_id} is already running!")
            return
        
        # Check if client data exists
        client_file = f"data/processed/client{client_id}_data.csv"
        if not os.path.exists(client_file):
            messagebox.showerror("Error", f"Client {client_id} data not found!\nRun: python split_client_data.py")
            return
        
        self.log_client(client_id, f"[START] Client {client_id}...")
        
        # Start client process
        cmd = [sys.executable, "client_autoencoder.py", str(client_id)]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        self.client_processes[client_id] = process
        
        # Start thread to read output
        threading.Thread(
            target=self.read_process_output,
            args=(process, self.client_queues[client_id], f"CLIENT_{client_id}"),
            daemon=True
        ).start()
        
        # Update UI
        self.client_statuses[client_id] = "Running"
        self.update_client_status(client_id)
        self.client_start_btns[client_id].config(state=tk.DISABLED)
        self.client_stop_btns[client_id].config(state=tk.NORMAL)
        
        self.log_all(f"[OK] Client {client_id} started", "success")
    
    def stop_client(self, client_id):
        """Stop a specific client"""
        if client_id not in self.client_processes:
            return
        
        self.log_client(client_id, f"[STOP] Client {client_id}...")
        
        try:
            self.client_processes[client_id].terminate()
            self.client_processes[client_id].wait(timeout=5)
        except:
            self.client_processes[client_id].kill()
        
        del self.client_processes[client_id]
        self.client_statuses[client_id] = "Stopped"
        self.update_client_status(client_id)
        
        self.client_start_btns[client_id].config(state=tk.NORMAL)
        self.client_stop_btns[client_id].config(state=tk.DISABLED)
        
        self.log_all(f"[OK] Client {client_id} stopped", "info")
    
    def start_full_system(self):
        """Start server and all clients"""
        # Start server first
        if self.server_process is None:
            self.start_server()
        
        # Wait 5 seconds for server to actually start listening (increased from 2)
        self.log_all("[WAIT] Waiting 5 seconds for server to start listening...", "info")
        self.root.after(5000, self._start_all_clients)  # Changed from 2000 to 5000
    
    def _start_all_clients(self):
        """Helper to start all clients with delay"""
        self.log_all("[START] Starting clients now...", "info")
        for i in range(1, self.max_clients + 1):
            if i not in self.client_processes:
                if os.path.exists(f"data/processed/client{i}_data.csv"):
                    # Start clients with 2 second delay between each
                    self.root.after(2000 * (i-1), lambda client_id=i: self.start_client(client_id))

    
    def stop_all(self):
        """Stop all processes"""
        # Stop all clients
        for client_id in list(self.client_processes.keys()):
            self.stop_client(client_id)
        
        # Stop server
        if self.server_process is not None:
            self.stop_server()
        
        self.log_all("[STOP] All processes stopped", "info")
    
    def read_process_output(self, process, output_queue, prefix):
        """Read process output in separate thread"""
        try:
            for line in iter(process.stdout.readline, ''):
                if line:
                    output_queue.put((prefix, line.strip()))
        except:
            pass
        finally:
            process.stdout.close()
    
    def process_queues(self):
        """Process output queues and update UI"""
        # Server output
        try:
            while True:
                item = self.server_queue.get_nowait()
                if isinstance(item, tuple) and len(item) == 2:
                    _prefix, line = item
                else:
                    line = str(item)

                self._parse_async_line(line)   # <-- add this
                self.log_server(line)
                self.log_all(f"[SERVER] {line}", "server")
        except queue.Empty:
            pass
        
        # Client outputs
        for client_id in range(1, self.max_clients + 1):
            try:
                while True:
                    prefix, line = self.client_queues[client_id].get_nowait()
                    self.log_client(client_id, line)
                    self.log_all(f"[CLIENT {client_id}] {line}", "client")
            except queue.Empty:
                pass
        
        # Schedule next check
        self.root.after(100, self.process_queues)
    
    def _parse_async_line(self, line: str):
        """Parse lines like: [ASYNC] fedbuff=1 version=3 buffered=2 applied=1 avg_staleness=0.5000"""
        if "[ASYNC]" not in line:
            return

        pairs = dict(re.findall(r"(\w+)=([^\s]+)", line))
        if not pairs:
            return

        self.async_stats["fedbuff"] = "Enabled" if pairs.get("fedbuff", "0") == "1" else "Disabled"
        self.async_stats["version"] = pairs.get("version", self.async_stats["version"])
        self.async_stats["buffered"] = pairs.get("buffered", self.async_stats["buffered"])
        self.async_stats["applied"] = pairs.get("applied", self.async_stats["applied"])
        self.async_stats["avg_staleness"] = pairs.get("avg_staleness", self.async_stats["avg_staleness"])

        try:
            applied = float(self.async_stats["applied"])
            staleness = float(self.async_stats["avg_staleness"])
            if applied > 0:
                self.async_stats["status"] = "Active"
                status_color = self.colors["success"]
            elif staleness > 0:
                self.async_stats["status"] = "Buffered/Stale"
                status_color = self.colors["warning"]
            else:
                self.async_stats["status"] = "Idle"
                status_color = self.colors["text_muted"]
        except Exception:
            self.async_stats["status"] = "Unknown"
            status_color = self.colors["text_muted"]

        for key, lbl in self.async_labels.items():
            lbl.config(text=self.async_stats[key])
        if "status" in self.async_labels:
            self.async_labels["status"].config(fg=status_color)

    def log_server(self, message):
        """Log message to server output"""
        self.server_output.insert(tk.END, message + "\n")
        self.server_output.see(tk.END)
    
    def log_client(self, client_id, message):
        """Log message to client output"""
        if client_id in self.client_outputs:
            self.client_outputs[client_id].insert(tk.END, message + "\n")
            self.client_outputs[client_id].see(tk.END)
    
    def log_all(self, message, level="info"):
        """Log message to all logs tab"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Color coding
        colors = {
            "success": self.colors["success"],
            "error": self.colors["danger"],
            "warning": self.colors["warning"],
            "info": self.colors["info"],
            "server": self.colors["purple"],
            "client": self.colors["teal"]
        }
        
        self.all_logs.insert(tk.END, f"[{timestamp}] ", "timestamp")
        self.all_logs.insert(tk.END, f"{message}\n", level)
        
        # Apply color tags
        self.all_logs.tag_config("timestamp", foreground=self.colors["text_muted"])
        for tag, color in colors.items():
            self.all_logs.tag_config(tag, foreground=color)
        
        self.all_logs.see(tk.END)
    
    def update_server_status(self):
        """Update server status display"""
        if self.server_status == "Running":
            self.server_status_label.config(text="● Running", fg=self.colors["success"])
        else:
            self.server_status_label.config(text="● Stopped", fg=self.colors["danger"])
    
    def update_client_status(self, client_id):
        """Update client status display"""
        if self.client_statuses[client_id] == "Running":
            self.client_status_labels[client_id].config(text="● Running", fg=self.colors["success"])
        else:
            self.client_status_labels[client_id].config(text="● Stopped", fg=self.colors["text_muted"])
    
    def generate_results(self):
        """Generate metrics and visualizations"""
        self.log_all("[INFO] Generating results...", "info")
        
        try:
            result = subprocess.run(
                [sys.executable, "generate_metrics.py"],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                self.log_all("[OK] Results generated successfully!", "success")
                self.load_results()
                messagebox.showinfo("Success", "Metrics generated successfully!\nCheck the 'Results' tab.")
            else:
                self.log_all(f"[ERROR] Error generating results:\n{result.stderr}", "error")
                messagebox.showerror("Error", f"Failed to generate results:\n{result.stderr[:500]}")
        except subprocess.TimeoutExpired:
            self.log_all("[WARNING] Timeout while generating results", "warning")
            messagebox.showwarning("Timeout", "Result generation timed out.")
        except Exception as e:
            self.log_all(f"[ERROR] Error: {e}", "error")
            messagebox.showerror("Error", str(e))
    
    def load_results(self):
        """Load and display latest results"""
        try:
            result_files = [f for f in os.listdir("results") if f.endswith(".json")]
            if not result_files:
                self.results_text.delete(1.0, tk.END)
                self.results_text.insert(tk.END, "No results found.\nRun training first.")
                return
            
            latest = max(result_files, key=lambda f: os.path.getctime(os.path.join("results", f)))
            filepath = os.path.join("results", latest)
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Display summary
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"[RESULTS] Latest Results: {latest}\n\n", "header")
            
            self.results_text.insert(tk.END, f"Experiment ID: {data.get('experiment_id', 'N/A')}\n")
            self.results_text.insert(tk.END, f"Model Type: {data.get('model_type', 'N/A')}\n")
            self.results_text.insert(tk.END, f"Total Rounds: {len(data.get('rounds', []))}\n")
            self.results_text.insert(tk.END, f"Clients: {data.get('num_clients', 'N/A')}\n\n")
            
            self.results_text.insert(tk.END, "=" * 60 + "\n")
            self.results_text.insert(tk.END, "FINAL ROUND METRICS\n")
            self.results_text.insert(tk.END, "=" * 60 + "\n\n")
            
            if data.get('rounds'):
                last_round = data['rounds'][-1]
                metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc', 'specificity']
                
                for metric in metrics:
                    if metric in last_round:
                        self.results_text.insert(tk.END, f"{metric.replace('_', ' ').title():<20}: {last_round[metric]:.4f}\n")
            
            self.results_text.tag_config("header", font=("Segoe UI", 12, "bold"), foreground=self.colors["info"])
            
        except Exception as e:
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Error loading results: {e}")
    
    def open_results_folder(self):
        """Open results folder in file explorer"""
        if os.path.exists("results"):
            if sys.platform == "win32":
                os.startfile("results")
            elif sys.platform == "darwin":
                subprocess.run(["open", "results"])
            else:
                subprocess.run(["xdg-open", "results"])
        else:
            messagebox.showwarning("Warning", "Results folder not found.")
    
    def on_closing(self):
        """Handle window close"""
        if messagebox.askokcancel("Quit", "Stop all processes and quit?"):
            self.stop_all()
            self.root.destroy()


def main():
    root = tk.Tk()
    app = FederatedLearningDashboard(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()