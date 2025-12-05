"""
Federated Learning Dashboard - GUI for Server and Client Management
Provides a unified interface to start/stop server and clients
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import subprocess
import threading
import queue
import os
import sys
import json
from datetime import datetime


class FederatedLearningDashboard:
    """Main GUI Dashboard for Federated Learning"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Federated Learning Dashboard - IoT Anomaly Detection")
        self.root.geometry("1400x900")
        self.root.configure(bg='#2C3E50')
        
        # Process tracking
        self.server_process = None
        self.client_processes = {}  # {client_id: process}
        self.max_clients = 5
        
        # Output queues for threading
        self.server_queue = queue.Queue()
        self.client_queues = {i: queue.Queue() for i in range(1, self.max_clients + 1)}
        
        # Status tracking
        self.server_status = "Stopped"
        self.client_statuses = {i: "Stopped" for i in range(1, self.max_clients + 1)}
        
        self.setup_ui()
        self.check_prerequisites()
        
        # Start queue monitoring
        self.root.after(100, self.process_queues)
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main container with padding
        main_container = tk.Frame(self.root, bg='#2C3E50')
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_frame = tk.Frame(main_container, bg='#34495E', relief=tk.RAISED, bd=2)
        title_frame.pack(fill=tk.X, pady=(0, 10))
        
        title = tk.Label(
            title_frame,
            text="🚀 Federated Learning Control Center",
            font=("Arial", 20, "bold"),
            bg='#34495E',
            fg='#ECF0F1',
            pady=15
        )
        title.pack()
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_container)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Style configuration
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TNotebook', background='#2C3E50')
        style.configure('TNotebook.Tab', padding=[20, 10], font=('Arial', 10, 'bold'))
        
        # Create tabs
        self.setup_overview_tab()
        self.setup_server_tab()
        self.setup_clients_tab()
        self.setup_logs_tab()
        self.setup_results_tab()
    
    def setup_overview_tab(self):
        """Overview tab with system status"""
        overview_frame = tk.Frame(self.notebook, bg='#34495E')
        self.notebook.add(overview_frame, text="📊 Overview")
        
        # Status panel
        status_panel = tk.LabelFrame(
            overview_frame,
            text="System Status",
            font=("Arial", 14, "bold"),
            bg='#34495E',
            fg='#ECF0F1',
            padx=20,
            pady=20
        )
        status_panel.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Server status
        server_frame = tk.Frame(status_panel, bg='#34495E')
        server_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(
            server_frame,
            text="Server:",
            font=("Arial", 12, "bold"),
            bg='#34495E',
            fg='#ECF0F1',
            width=15,
            anchor='w'
        ).pack(side=tk.LEFT, padx=5)
        
        self.server_status_label = tk.Label(
            server_frame,
            text="● Stopped",
            font=("Arial", 12),
            bg='#34495E',
            fg='#E74C3C',
            width=20,
            anchor='w'
        )
        self.server_status_label.pack(side=tk.LEFT, padx=5)
        
        # Client statuses
        self.client_status_labels = {}
        for i in range(1, self.max_clients + 1):
            client_frame = tk.Frame(status_panel, bg='#34495E')
            client_frame.pack(fill=tk.X, pady=5)
            
            tk.Label(
                client_frame,
                text=f"Client {i}:",
                font=("Arial", 12),
                bg='#34495E',
                fg='#ECF0F1',
                width=15,
                anchor='w'
            ).pack(side=tk.LEFT, padx=5)
            
            status_label = tk.Label(
                client_frame,
                text="● Stopped",
                font=("Arial", 12),
                bg='#34495E',
                fg='#95A5A6',
                width=20,
                anchor='w'
            )
            status_label.pack(side=tk.LEFT, padx=5)
            self.client_status_labels[i] = status_label
        
        # Quick actions
        actions_frame = tk.LabelFrame(
            overview_frame,
            text="Quick Actions",
            font=("Arial", 14, "bold"),
            bg='#34495E',
            fg='#ECF0F1',
            padx=20,
            pady=20
        )
        actions_frame.pack(fill=tk.X, padx=10, pady=10)
        
        btn_frame = tk.Frame(actions_frame, bg='#34495E')
        btn_frame.pack()
        
        tk.Button(
            btn_frame,
            text="🚀 Start Full System",
            font=("Arial", 12, "bold"),
            bg='#27AE60',
            fg='white',
            padx=20,
            pady=10,
            command=self.start_full_system,
            cursor='hand2'
        ).pack(side=tk.LEFT, padx=10)
        
        tk.Button(
            btn_frame,
            text="🛑 Stop All",
            font=("Arial", 12, "bold"),
            bg='#E74C3C',
            fg='white',
            padx=20,
            pady=10,
            command=self.stop_all,
            cursor='hand2'
        ).pack(side=tk.LEFT, padx=10)
        
        tk.Button(
            btn_frame,
            text="📊 View Results",
            font=("Arial", 12, "bold"),
            bg='#3498DB',
            fg='white',
            padx=20,
            pady=10,
            command=self.generate_results,
            cursor='hand2'
        ).pack(side=tk.LEFT, padx=10)
    
    def setup_server_tab(self):
        """Server control tab"""
        server_frame = tk.Frame(self.notebook, bg='#34495E')
        self.notebook.add(server_frame, text="🖥️ Server")
        
        # Control panel
        control_panel = tk.LabelFrame(
            server_frame,
            text="Server Control",
            font=("Arial", 14, "bold"),
            bg='#34495E',
            fg='#ECF0F1',
            padx=20,
            pady=20
        )
        control_panel.pack(fill=tk.X, padx=10, pady=10)
        
        # Rounds input
        rounds_frame = tk.Frame(control_panel, bg='#34495E')
        rounds_frame.pack(pady=10)
        
        tk.Label(
            rounds_frame,
            text="Number of Rounds:",
            font=("Arial", 12),
            bg='#34495E',
            fg='#ECF0F1'
        ).pack(side=tk.LEFT, padx=5)
        
        self.rounds_var = tk.StringVar(value="10")
        rounds_entry = tk.Entry(
            rounds_frame,
            textvariable=self.rounds_var,
            font=("Arial", 12),
            width=10
        )
        rounds_entry.pack(side=tk.LEFT, padx=5)
        
        # Buttons
        btn_frame = tk.Frame(control_panel, bg='#34495E')
        btn_frame.pack(pady=10)
        
        self.start_server_btn = tk.Button(
            btn_frame,
            text="▶ Start Server",
            font=("Arial", 12, "bold"),
            bg='#27AE60',
            fg='white',
            padx=30,
            pady=10,
            command=self.start_server,
            cursor='hand2'
        )
        self.start_server_btn.pack(side=tk.LEFT, padx=10)
        
        self.stop_server_btn = tk.Button(
            btn_frame,
            text="⏹ Stop Server",
            font=("Arial", 12, "bold"),
            bg='#E74C3C',
            fg='white',
            padx=30,
            pady=10,
            command=self.stop_server,
            state=tk.DISABLED,
            cursor='hand2'
        )
        self.stop_server_btn.pack(side=tk.LEFT, padx=10)
        
        # Server output
        output_panel = tk.LabelFrame(
            server_frame,
            text="Server Output",
            font=("Arial", 12, "bold"),
            bg='#34495E',
            fg='#ECF0F1',
            padx=10,
            pady=10
        )
        output_panel.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.server_output = scrolledtext.ScrolledText(
            output_panel,
            font=("Consolas", 10),
            bg='#1C2833',
            fg='#00FF00',
            insertbackground='white',
            wrap=tk.WORD
        )
        self.server_output.pack(fill=tk.BOTH, expand=True)
    
    def setup_clients_tab(self):
        """Clients control tab"""
        clients_frame = tk.Frame(self.notebook, bg='#34495E')
        self.notebook.add(clients_frame, text="👥 Clients")
        
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
        client_frame = tk.Frame(self.client_notebook, bg='#34495E')
        self.client_notebook.add(client_frame, text=f"Client {client_id}")
        self.client_tabs[client_id] = client_frame
        
        # Control panel
        control_panel = tk.LabelFrame(
            client_frame,
            text=f"Client {client_id} Control",
            font=("Arial", 12, "bold"),
            bg='#34495E',
            fg='#ECF0F1',
            padx=20,
            pady=15
        )
        control_panel.pack(fill=tk.X, padx=10, pady=10)
        
        btn_frame = tk.Frame(control_panel, bg='#34495E')
        btn_frame.pack()
        
        start_btn = tk.Button(
            btn_frame,
            text=f"▶ Start Client {client_id}",
            font=("Arial", 11, "bold"),
            bg='#27AE60',
            fg='white',
            padx=25,
            pady=8,
            command=lambda: self.start_client(client_id),
            cursor='hand2'
        )
        start_btn.pack(side=tk.LEFT, padx=10)
        self.client_start_btns[client_id] = start_btn
        
        stop_btn = tk.Button(
            btn_frame,
            text=f"⏹ Stop Client {client_id}",
            font=("Arial", 11, "bold"),
            bg='#E74C3C',
            fg='white',
            padx=25,
            pady=8,
            command=lambda: self.stop_client(client_id),
            state=tk.DISABLED,
            cursor='hand2'
        )
        stop_btn.pack(side=tk.LEFT, padx=10)
        self.client_stop_btns[client_id] = stop_btn
        
        # Output
        output_panel = tk.LabelFrame(
            client_frame,
            text=f"Client {client_id} Output",
            font=("Arial", 11, "bold"),
            bg='#34495E',
            fg='#ECF0F1',
            padx=10,
            pady=10
        )
        output_panel.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        output = scrolledtext.ScrolledText(
            output_panel,
            font=("Consolas", 9),
            bg='#1C2833',
            fg='#3498DB',
            insertbackground='white',
            wrap=tk.WORD
        )
        output.pack(fill=tk.BOTH, expand=True)
        self.client_outputs[client_id] = output
    
    def setup_logs_tab(self):
        """Combined logs tab"""
        logs_frame = tk.Frame(self.notebook, bg='#34495E')
        self.notebook.add(logs_frame, text="📝 All Logs")
        
        self.all_logs = scrolledtext.ScrolledText(
            logs_frame,
            font=("Consolas", 9),
            bg='#1C2833',
            fg='#ECF0F1',
            insertbackground='white',
            wrap=tk.WORD
        )
        self.all_logs.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def setup_results_tab(self):
        """Results visualization tab"""
        results_frame = tk.Frame(self.notebook, bg='#34495E')
        self.notebook.add(results_frame, text="📈 Results")
        
        control_frame = tk.Frame(results_frame, bg='#34495E')
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Button(
            control_frame,
            text="🔄 Refresh Results",
            font=("Arial", 11, "bold"),
            bg='#3498DB',
            fg='white',
            padx=20,
            pady=8,
            command=self.load_results,
            cursor='hand2'
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            control_frame,
            text="📊 Generate Metrics",
            font=("Arial", 11, "bold"),
            bg='#9B59B6',
            fg='white',
            padx=20,
            pady=8,
            command=self.generate_results,
            cursor='hand2'
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            control_frame,
            text="📁 Open Results Folder",
            font=("Arial", 11, "bold"),
            bg='#16A085',
            fg='white',
            padx=20,
            pady=8,
            command=self.open_results_folder,
            cursor='hand2'
        ).pack(side=tk.LEFT, padx=5)
        
        # Results display
        self.results_text = scrolledtext.ScrolledText(
            results_frame,
            font=("Consolas", 10),
            bg='#1C2833',
            fg='#ECF0F1',
            wrap=tk.WORD
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
            self.log_all(f"⚠️ WARNING: Missing files:\n" + "\n".join(f"  - {f}" for f in missing), "warning")
        else:
            self.log_all("✅ All prerequisites found!", "success")
    
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
        
        self.log_server("🚀 Starting Federated Learning Server...")
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
        
        self.log_all("🖥️ Server started successfully", "success")
    
    def stop_server(self):
        """Stop the server"""
        if self.server_process is None:
            return
        
        self.log_server("🛑 Stopping server...")
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
        
        self.log_all("🖥️ Server stopped", "info")
    
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
        
        self.log_client(client_id, f"🚀 Starting Client {client_id}...")
        
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
        
        self.log_all(f"👤 Client {client_id} started", "success")
    
    def stop_client(self, client_id):
        """Stop a specific client"""
        if client_id not in self.client_processes:
            return
        
        self.log_client(client_id, f"🛑 Stopping Client {client_id}...")
        
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
        
        self.log_all(f"👤 Client {client_id} stopped", "info")
    
    def start_full_system(self):
        """Start server and all clients"""
        # Start server first
        if self.server_process is None:
            self.start_server()
        
        # Wait 5 seconds for server to actually start listening (increased from 2)
        self.log_all("⏳ Waiting 5 seconds for server to start listening...", "info")
        self.root.after(5000, self._start_all_clients)  # Changed from 2000 to 5000
    
    def _start_all_clients(self):
        """Helper to start all clients with delay"""
        self.log_all("🚀 Starting clients now...", "info")
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
        
        self.log_all("🛑 All processes stopped", "info")
    
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
                prefix, line = self.server_queue.get_nowait()
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
            "success": "#27AE60",
            "error": "#E74C3C",
            "warning": "#F39C12",
            "info": "#3498DB",
            "server": "#9B59B6",
            "client": "#1ABC9C"
        }
        
        self.all_logs.insert(tk.END, f"[{timestamp}] ", "timestamp")
        self.all_logs.insert(tk.END, f"{message}\n", level)
        
        # Apply color tags
        self.all_logs.tag_config("timestamp", foreground="#95A5A6")
        for tag, color in colors.items():
            self.all_logs.tag_config(tag, foreground=color)
        
        self.all_logs.see(tk.END)
    
    def update_server_status(self):
        """Update server status display"""
        if self.server_status == "Running":
            self.server_status_label.config(text="● Running", fg='#27AE60')
        else:
            self.server_status_label.config(text="● Stopped", fg='#E74C3C')
    
    def update_client_status(self, client_id):
        """Update client status display"""
        if self.client_statuses[client_id] == "Running":
            self.client_status_labels[client_id].config(text="● Running", fg='#27AE60')
        else:
            self.client_status_labels[client_id].config(text="● Stopped", fg='#95A5A6')
    
    def generate_results(self):
        """Generate metrics and visualizations"""
        self.log_all("📊 Generating results...", "info")
        
        try:
            result = subprocess.run(
                [sys.executable, "generate_metrics.py"],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                self.log_all("✅ Results generated successfully!", "success")
                self.load_results()
                messagebox.showinfo("Success", "Metrics generated successfully!\nCheck the 'Results' tab.")
            else:
                self.log_all(f"❌ Error generating results:\n{result.stderr}", "error")
                messagebox.showerror("Error", f"Failed to generate results:\n{result.stderr[:500]}")
        except subprocess.TimeoutExpired:
            self.log_all("⏱️ Timeout while generating results", "warning")
            messagebox.showwarning("Timeout", "Result generation timed out.")
        except Exception as e:
            self.log_all(f"❌ Error: {e}", "error")
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
            self.results_text.insert(tk.END, f"📄 Latest Results: {latest}\n\n", "header")
            
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
            
            self.results_text.tag_config("header", font=("Arial", 12, "bold"), foreground="#3498DB")
            
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