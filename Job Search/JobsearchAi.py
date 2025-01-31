import json
import os
from datetime import datetime
from typing import List, Dict
import PyPDF2
import requests
from dataclasses import dataclass
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from groq import Groq
import asyncio
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Groq client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set. Please set it with your Groq API key.")
groq_client = Groq(api_key=GROQ_API_KEY)

@dataclass
class JobPosting:
    title: str
    company: str
    location: str
    description: str
    required_skills: List[str]
    salary_range: str
    posting_date: str

class JobSearchUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Job Search Assistant (Powered by Groq)")
        self.root.geometry("1000x800")
        self.assistant = JobSearchAssistant()
        
        # Configure style
        self.setup_styles()
        self.create_widgets()
        
    def setup_styles(self):
        """Setup custom styles for widgets"""
        style = ttk.Style()
        
        # Configure colors
        self.bg_color = "#f9f9f9"
        self.accent_color = "#007bff"
        self.text_color = "#333"
        self.secondary_color = "#6c757d"
        
        # Configure base style
        style.configure(".", 
                      font=("Helvetica", 11),
                      background=self.bg_color,
                      foreground=self.text_color)
        
        # Configure labels
        style.configure("Title.TLabel",
                      font=("Helvetica", 26, "bold"),
                      padding=12,
                      foreground=self.accent_color)
        
        style.configure("Subtitle.TLabel",
                      font=("Helvetica", 12, "italic"),
                      padding=5,
                      foreground=self.secondary_color)
        
        style.configure("Header.TLabel",
                      font=("Helvetica", 14, "bold"),
                      padding=5,
                      foreground=self.text_color)
        
        # Configure frames
        style.configure("Card.TFrame",
                      padding=20,
                      relief="raised",
                      borderwidth=2,
                      background="white")
        
        style.configure("MainFrame.TFrame",
                      padding=20,
                      background=self.bg_color)
        
        # Configure buttons
        style.configure("Primary.TButton",
                      font=("Helvetica", 10, "bold"),
                      padding=12,
                      background=self.accent_color,
                      foreground="white")
        style.map("Primary.TButton",
                  background=[("active", "#0056b3"), ("disabled", "#cfe2ff")])
        
        style.configure("Secondary.TButton",
                      font=("Helvetica", 10),
                      padding=10,
                      relief="flat",
                      background=self.secondary_color,
                      foreground="white")
        
        # Configure entries
        style.configure("TEntry",
                      padding=5,
                      relief="solid",
                      borderwidth=2,
                      background="white")

    def create_widgets(self):
        """Create and arrange UI widgets"""
        # Configure root
        self.root.configure(bg=self.bg_color)
        
        # Main container with padding
        main_frame = ttk.Frame(self.root, style="MainFrame.TFrame")
        main_frame.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
        
        # Configure grid
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)
        
        # Title
        title_frame = ttk.Frame(main_frame, style="MainFrame.TFrame")
        title_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 20))
        
        title_label = ttk.Label(title_frame, text="Job Search Assistant", style="Title.TLabel")
        title_label.grid(row=0, column=0, sticky="w")
        
        subtitle_label = ttk.Label(title_frame, text="Powered by Groq AI", style="Subtitle.TLabel")
        subtitle_label.grid(row=1, column=0, sticky="w")
        
        # Left panel (Input fields)
        left_panel = ttk.Frame(main_frame, style="MainFrame.TFrame")
        left_panel.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        
        # Resume section
        resume_frame = ttk.LabelFrame(left_panel, text="Resume Upload", style="Card.TLabelframe")
        resume_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        resume_frame.grid_columnconfigure(0, weight=1)
        
        self.resume_path_var = tk.StringVar()
        browse_button = ttk.Button(resume_frame, text="📄 Upload Resume (PDF)", command=self.browse_resume, style="Primary.TButton")
        browse_button.grid(row=0, column=0, sticky="ew", pady=5)
        
        self.resume_label = ttk.Label(resume_frame, text="No resume selected", wraplength=300)
        self.resume_label.grid(row=1, column=0, sticky="w")
        
        # Profile Display
        profile_frame = ttk.LabelFrame(left_panel, text="Your Profile", style="Card.TLabelframe")
        profile_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        profile_frame.grid_columnconfigure(0, weight=1)
        
        self.profile_text = tk.Text(profile_frame, wrap=tk.WORD, height=8, font=("Helvetica", 10))
        self.profile_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.profile_text.configure(state="disabled")
        
        # Search preferences section
        pref_frame = ttk.LabelFrame(left_panel, text="Search Preferences", style="Card.TLabelframe")
        pref_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        pref_frame.grid_columnconfigure(1, weight=1)
        
        # Job search
        ttk.Label(pref_frame, text="🔍 Job Title:", style="Header.TLabel").grid(row=0, column=0, sticky="w", pady=5)
        self.job_search_var = tk.StringVar()
        ttk.Entry(pref_frame, textvariable=self.job_search_var, width=30).grid(row=0, column=1, sticky="ew", padx=5)
        
        # Locations
        ttk.Label(pref_frame, text="📍 Locations:", style="Header.TLabel").grid(row=1, column=0, sticky="w", pady=5)
        self.locations_var = tk.StringVar()
        ttk.Entry(pref_frame, textvariable=self.locations_var, width=30).grid(row=1, column=1, sticky="ew", padx=5)
        ttk.Label(pref_frame, text="Separate multiple locations with commas", style="Subtitle.TLabel").grid(row=2, column=1, sticky="w", padx=5)
        
        # Minimum salary
        ttk.Label(pref_frame, text="💰 Min Salary:", style="Header.TLabel").grid(row=3, column=0, sticky="w", pady=5)
        self.min_salary_var = tk.StringVar(value="0")
        ttk.Entry(pref_frame, textvariable=self.min_salary_var, width=30).grid(row=3, column=1, sticky="ew", padx=5)
        
        # Remote preference
        self.remote_var = tk.BooleanVar()
        ttk.Checkbutton(pref_frame, text="🏠 Remote Only", variable=self.remote_var).grid(row=4, column=0, columnspan=2, sticky="w", pady=5)
        
        # Search button
        search_button = ttk.Button(left_panel, text="🔍 Search Jobs", command=self.search_jobs, style="Primary.TButton")
        search_button.grid(row=3, column=0, sticky="ew", pady=10)
        
        # Right panel (Results)
        right_panel = ttk.Frame(main_frame, style="Card.TFrame")
        right_panel.grid(row=1, column=1, sticky="nsew")
        right_panel.grid_columnconfigure(0, weight=1)
        right_panel.grid_rowconfigure(1, weight=1)
        
        # Results header
        ttk.Label(right_panel, text="Job Recommendations", style="Header.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 10))
        
        # Results text area with scrollbar
        results_frame = ttk.Frame(right_panel, style="Card.TFrame")
        results_frame.grid(row=1, column=0, sticky="nsew")
        results_frame.grid_columnconfigure(0, weight=1)
        results_frame.grid_rowconfigure(0, weight=1)
        
        self.results_text = tk.Text(results_frame, wrap=tk.WORD, width=50, height=30)
        self.results_text.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.results_text.configure(font=("Helvetica", 10))
        
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.results_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.results_text.configure(yscrollcommand=scrollbar.set)

def main():
    root = tk.Tk()
    app = JobSearchUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
