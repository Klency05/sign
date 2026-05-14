# Integrated Sign Language Recognition + Medical RAG Application

import numpy as np
import cv2
import os, sys, time, operator
from string import ascii_uppercase
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from PIL import Image, ImageTk
from spellchecker import SpellChecker
from keras.models import model_from_json
from sentence_transformers import SentenceTransformer
import chromadb
import ollama
import threading

class IntegratedApplication:

    def __init__(self):
        # Initialize spell checker and video
        self.spell = SpellChecker()
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.current_image2 = None
        
        # Load sign language models
        self._load_sign_models()
        
        # Initialize counters
        self.ct = {}
        self.ct['blank'] = 0
        self.blank_flag = 0
        for i in ascii_uppercase:
            self.ct[i] = 0
        
        # Initialize recognized text variables
        self.str = ""
        self.word = ""
        self.current_symbol = "Empty"
        
        # Load RAG components
        print("Loading RAG components...")
        self._load_rag_components()
        
        # Create GUI
        self.root = tk.Tk()
        self.root.title("Sign Language & Medical RAG Assistant")
        self.root.geometry("1200x900")
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)
        
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create tabs
        self.sign_frame = ttk.Frame(self.notebook)
        self.rag_frame = ttk.Frame(self.notebook)
        
        self.notebook.add(self.sign_frame, text="Sign Language Recognition")
        self.notebook.add(self.rag_frame, text="Medical RAG Assistant")
        
        # Setup sign language tab
        self._setup_sign_tab()
        
        # Setup RAG tab
        self._setup_rag_tab()
        
        # Start video loop for sign recognition
        self.video_loop()
        
    def _load_sign_models(self):
        """Load all sign language recognition models"""
        print("Loading sign language models...")
        
        # Get script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Load main model
        model_path = os.path.join(script_dir, "Models", "model_new.json")
        with open(model_path, "r") as f:
            self.model_json = f.read()
        self.loaded_model = model_from_json(self.model_json)
        self.loaded_model.load_weights(os.path.join(script_dir, "Models", "model_new.h5"))
        
        # Load specialized models
        model_dru_path = os.path.join(script_dir, "Models", "model-bw_dru.json")
        with open(model_dru_path, "r") as f:
            self.model_json_dru = f.read()
        self.loaded_model_dru = model_from_json(self.model_json_dru)
        self.loaded_model_dru.load_weights(os.path.join(script_dir, "Models", "model-bw_dru.h5"))
        
        model_tkdi_path = os.path.join(script_dir, "Models", "model-bw_tkdi.json")
        with open(model_tkdi_path, "r") as f:
            self.model_json_tkdi = f.read()
        self.loaded_model_tkdi = model_from_json(self.model_json_tkdi)
        self.loaded_model_tkdi.load_weights(os.path.join(script_dir, "Models", "model-bw_tkdi.h5"))
        
        model_smn_path = os.path.join(script_dir, "Models", "model-bw_smn.json")
        with open(model_smn_path, "r") as f:
            self.model_json_smn = f.read()
        self.loaded_model_smn = model_from_json(self.model_json_smn)
        self.loaded_model_smn.load_weights(os.path.join(script_dir, "Models", "model-bw_smn.h5"))
        
        print("Sign models loaded successfully!")
    
    def _load_rag_components(self):
        """Load RAG components"""
        try:
            # Get script directory
            script_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(script_dir)
            
            # Load embedding model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Load ChromaDB - look in medical_rag directory
            medical_db_path = os.path.join(parent_dir, "medical-rag", "medical_db")
            self.client_db = chromadb.PersistentClient(path=medical_db_path)
            self.collection = self.client_db.get_collection(name="medical_qa")
            
            self.rag_ready = True
            print("RAG components loaded successfully!")
        except Exception as e:
            self.rag_ready = False
            print(f"Warning: Could not load RAG components: {e}")
    
    def _setup_sign_tab(self):
        """Setup sign language recognition tab"""
        # Title
        title = tk.Label(self.sign_frame, text="Sign Language to Text Conversion", 
                        font=("Courier", 20, "bold"))
        title.pack(pady=5)
        
        # Video feed
        self.panel = tk.Label(self.sign_frame)
        self.panel.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Right panel for info
        right_frame = tk.Frame(self.sign_frame)
        right_frame.pack(side=tk.RIGHT, padx=10, pady=5, fill=tk.BOTH, expand=True)
        
        # Processed image
        self.panel2 = tk.Label(right_frame, text="Processed Image", bg="gray")
        self.panel2.pack(pady=5)
        
        # Current symbol
        char_frame = tk.Frame(right_frame)
        char_frame.pack(pady=5)
        tk.Label(char_frame, text="Character:", font=("Courier", 12, "bold")).pack(side=tk.LEFT)
        self.panel3 = tk.Label(char_frame, text="", font=("Courier", 12))
        self.panel3.pack(side=tk.LEFT, padx=5)
        
        # Word
        word_frame = tk.Frame(right_frame)
        word_frame.pack(pady=5)
        tk.Label(word_frame, text="Word:", font=("Courier", 12, "bold")).pack(side=tk.LEFT)
        self.panel4 = tk.Label(word_frame, text="", font=("Courier", 12))
        self.panel4.pack(side=tk.LEFT, padx=5)
        
        # Sentence
        sent_frame = tk.Frame(right_frame)
        sent_frame.pack(pady=5, fill=tk.BOTH, expand=True)
        tk.Label(sent_frame, text="Sentence:", font=("Courier", 12, "bold")).pack(anchor="nw")
        self.panel5 = tk.Label(sent_frame, text="", font=("Courier", 12), 
                              bg="white", wraplength=300, justify=tk.LEFT)
        self.panel5.pack(side=tk.LEFT, padx=5, fill=tk.BOTH, expand=True)
        
        # Suggestions
        suggestions_frame = tk.Frame(right_frame)
        suggestions_frame.pack(pady=5, fill=tk.BOTH)
        tk.Label(suggestions_frame, text="Suggestions:", font=("Courier", 10, "bold")).pack(anchor="nw")
        
        self.suggestion_buttons = []
        for i in range(3):
            btn = tk.Button(suggestions_frame, font=("Courier", 9), width=20,
                           command=lambda idx=i: self._apply_suggestion(idx))
            btn.pack(anchor="w", pady=2)
            self.suggestion_buttons.append(btn)
        
        # Query RAG button
        query_frame = tk.Frame(right_frame)
        query_frame.pack(pady=10, fill=tk.X)
        self.query_rag_btn = tk.Button(query_frame, text="Query Medical RAG with recognized text",
                                      command=self._query_rag_with_text, 
                                      font=("Courier", 9), bg="lightblue")
        self.query_rag_btn.pack(fill=tk.X)
        
        # Clear button
        clear_frame = tk.Frame(right_frame)
        clear_frame.pack(pady=5, fill=tk.X)
        tk.Button(clear_frame, text="Clear Text", command=self._clear_text,
                 font=("Courier", 9)).pack(fill=tk.X)
    
    def _setup_rag_tab(self):
        """Setup Medical RAG tab"""
        # Title
        title = tk.Label(self.rag_frame, text="Medical RAG Assistant", 
                        font=("Courier", 20, "bold"))
        title.pack(pady=5)
        
        # Main frame
        main_frame = tk.Frame(self.rag_frame)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Query input
        query_label = tk.Label(main_frame, text="Enter symptoms/medical question:",
                              font=("Courier", 10, "bold"))
        query_label.pack(anchor="w", pady=5)
        
        self.query_input = tk.Entry(main_frame, font=("Courier", 11), width=80)
        self.query_input.pack(fill=tk.X, pady=5)
        self.query_input.bind("<Return>", lambda e: self._search_medical_rag())
        
        # Button frame
        button_frame = tk.Frame(main_frame)
        button_frame.pack(pady=5, fill=tk.X)
        
        tk.Button(button_frame, text="Search", command=self._search_medical_rag,
                 font=("Courier", 10), bg="lightgreen").pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Use last recognized text", 
                 command=self._use_last_recognized, 
                 font=("Courier", 10), bg="lightblue").pack(side=tk.LEFT, padx=5)
        
        # Results frame
        results_label = tk.Label(main_frame, text="Medical Context Retrieved:",
                                font=("Courier", 10, "bold"))
        results_label.pack(anchor="w", pady=(15, 5))
        
        self.context_display = scrolledtext.ScrolledText(main_frame, height=10, 
                                                        font=("Courier", 9), 
                                                        wrap=tk.WORD)
        self.context_display.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # AI Response frame
        response_label = tk.Label(main_frame, text="AI Medical Response:",
                                 font=("Courier", 10, "bold"))
        response_label.pack(anchor="w", pady=(15, 5))
        
        self.response_display = scrolledtext.ScrolledText(main_frame, height=10,
                                                         font=("Courier", 9),
                                                         wrap=tk.WORD)
        self.response_display.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Status label
        self.status_label = tk.Label(main_frame, text="Ready", font=("Courier", 9))
        self.status_label.pack(anchor="w", pady=5)
    
    def video_loop(self):
        """Main video loop for sign recognition"""
        ok, frame = self.vs.read()
        
        if ok:
            cv2image = cv2.flip(frame, 1)
            
            # Define ROI
            x1 = int(0.5 * frame.shape[1])
            y1 = 10
            x2 = frame.shape[1] - 10
            y2 = int(0.5 * frame.shape[1])
            
            cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)
            cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)
            
            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)
            
            # Process ROI
            cv2image_roi = cv2image[y1:y2, x1:x2]
            gray = cv2.cvtColor(cv2image_roi, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 2)
            th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
            ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            self.predict(res)
            
            self.current_image2 = Image.fromarray(res)
            imgtk = ImageTk.PhotoImage(image=self.current_image2)
            self.panel2.imgtk = imgtk
            self.panel2.config(image=imgtk)
            
            # Update UI
            self.panel3.config(text=self.current_symbol)
            self.panel4.config(text=self.word)
            self.panel5.config(text=self.str)
            
            # Update suggestions
            candidates = self.spell.candidates(self.word)
            predicts = list(candidates) if candidates else []
            
            for i, btn in enumerate(self.suggestion_buttons):
                if i < len(predicts):
                    btn.config(text=predicts[i])
                else:
                    btn.config(text="")
        
        self.root.after(5, self.video_loop)
    
    def predict(self, test_image):
        """Predict gesture from image"""
        test_image = cv2.resize(test_image, (128, 128))
        
        # Main prediction
        result = self.loaded_model.predict(test_image.reshape(1, 128, 128, 1), verbose=0)
        result_dru = self.loaded_model_dru.predict(test_image.reshape(1, 128, 128, 1), verbose=0)
        result_tkdi = self.loaded_model_tkdi.predict(test_image.reshape(1, 128, 128, 1), verbose=0)
        result_smn = self.loaded_model_smn.predict(test_image.reshape(1, 128, 128, 1), verbose=0)
        
        prediction = {'blank': result[0][0]}
        
        for idx, i in enumerate(ascii_uppercase):
            prediction[i] = result[0][idx + 1]
        
        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        self.current_symbol = prediction[0][0]
        confidence = prediction[0][1]
        
        if confidence < 0.60:
            self.current_symbol = "blank"
            return
        
        # Layer 2 classification
        if self.current_symbol in ['D', 'R', 'U']:
            prediction = {'D': result_dru[0][0], 'R': result_dru[0][1], 'U': result_dru[0][2]}
            prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
            self.current_symbol = prediction[0][0]
        
        if self.current_symbol in ['D', 'I', 'K', 'T']:
            prediction = {'D': result_tkdi[0][0], 'I': result_tkdi[0][1], 
                         'K': result_tkdi[0][2], 'T': result_tkdi[0][3]}
            prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
            self.current_symbol = prediction[0][0]
        
        if self.current_symbol in ['M', 'N', 'S']:
            prediction1 = {'M': result_smn[0][0], 'N': result_smn[0][1], 'S': result_smn[0][2]}
            prediction1 = sorted(prediction1.items(), key=operator.itemgetter(1), reverse=True)
            self.current_symbol = prediction1[0][0]
        
        # Character counting
        if self.current_symbol == 'blank':
            for i in ascii_uppercase:
                self.ct[i] = 0
        
        self.ct[self.current_symbol] += 1
        
        # Wait for stable prediction
        if self.ct[self.current_symbol] > 40:
            for i in ascii_uppercase:
                if i == self.current_symbol:
                    continue
                tmp = self.ct[self.current_symbol] - self.ct[i]
                if abs(tmp) <= 15:
                    self.ct['blank'] = 0
                    for i in ascii_uppercase:
                        self.ct[i] = 0
                    return
            
            self.ct['blank'] = 0
            for i in ascii_uppercase:
                self.ct[i] = 0
            
            if self.current_symbol == 'blank':
                if self.blank_flag == 0:
                    self.blank_flag = 1
                    if len(self.str) > 0:
                        self.str += " "
                    self.str += self.word
                    self.word = ""
            else:
                if len(self.str) > 50:
                    self.str = ""
                self.blank_flag = 0
                self.word += self.current_symbol
    
    def _apply_suggestion(self, idx):
        """Apply spell correction suggestion"""
        candidates = self.spell.candidates(self.word)
        predicts = list(candidates) if candidates else []
        
        if idx < len(predicts):
            if len(self.str) > 0:
                self.str += " "
            self.str += predicts[idx]
            self.word = ""
    
    def _clear_text(self):
        """Clear all recognized text"""
        self.str = ""
        self.word = ""
    
    def _query_rag_with_text(self):
        """Send recognized text to RAG"""
        if not self.str.strip():
            messagebox.showwarning("Empty Input", "No recognized text to query")
            return
        
        self.query_input.delete(0, tk.END)
        self.query_input.insert(0, self.str)
        self.notebook.select(1)  # Switch to RAG tab
        self._search_medical_rag()
    
    def _use_last_recognized(self):
        """Fill input with last recognized text"""
        if self.str.strip():
            self.query_input.delete(0, tk.END)
            self.query_input.insert(0, self.str)
        else:
            messagebox.showwarning("No Text", "No recognized text available")
    
    def _search_medical_rag(self):
        """Search medical RAG with query"""
        if not self.rag_ready:
            messagebox.showerror("RAG Error", "RAG components not loaded. Check medical_db directory.")
            return
        
        query = self.query_input.get().strip()
        if not query:
            messagebox.showwarning("Empty Query", "Please enter a medical question")
            return
        
        self.status_label.config(text="Searching... please wait")
        self.root.update()
        
        # Run RAG in thread to avoid freezing UI
        thread = threading.Thread(target=self._run_rag_query, args=(query,))
        thread.daemon = True
        thread.start()
    
    def _run_rag_query(self, query):
        """Run RAG query in background"""
        try:
            # Encode query
            query_embedding = self.embedding_model.encode(query)
            
            # Retrieve similar documents
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=3
            )
            
            # Combine retrieved context
            retrieved_docs = "\n---\n".join(results['documents'][0])
            
            # Update context display
            self.context_display.config(state=tk.NORMAL)
            self.context_display.delete(1.0, tk.END)
            self.context_display.insert(1.0, retrieved_docs)
            self.context_display.config(state=tk.DISABLED)
            
            # Generate response with Mistral
            prompt = f"""You are a helpful medical assistant.

Use the medical context below to answer the user safely.

Medical Context:
{retrieved_docs}

User Question:
{query}

Rules:
- Do not provide dangerous diagnoses
- Suggest doctor consultation when needed
- Keep response concise and helpful
"""
            
            response = ollama.chat(
                model='mistral',
                messages=[{'role': 'user', 'content': prompt}]
            )
            
            # Update response display
            self.response_display.config(state=tk.NORMAL)
            self.response_display.delete(1.0, tk.END)
            self.response_display.insert(1.0, response['message']['content'])
            self.response_display.config(state=tk.DISABLED)
            
            self.status_label.config(text="Search completed successfully")
            
        except Exception as e:
            self.status_label.config(text=f"Error: {str(e)}")
            self.response_display.config(state=tk.NORMAL)
            self.response_display.delete(1.0, tk.END)
            self.response_display.insert(1.0, f"Error: {str(e)}")
            self.response_display.config(state=tk.DISABLED)
    
    def destructor(self):
        """Clean up and close application"""
        print("Closing Application...")
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print("Starting Integrated Application...")
    app = IntegratedApplication()
    app.root.mainloop()
