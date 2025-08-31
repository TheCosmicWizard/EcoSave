# EcoSave

***Sort Smart. Live Green.***

**Live Demo**: https://thecosmicwizard.github.io/EcoSave/

---

##  Overview

EcoSave is an **AI-powered waste recognition system** designed to make sustainability simple. It helps identify if waste is recyclable, compostable, or non-recyclable—empowering eco-conscious decisions for a cleaner planet.

---

##  User Experience

- **Intuitive Interface**: Clear calls-to-action like "Upload Waste Image" and "Analyze Waste."
- **Real-Time Feedback**: Users see status messages like “Connecting…”, “Loading…”, and placeholder guidance such as “Upload an image to see results”.
- **Visual Simplicity**: Clean design focused on purpose—identifying waste with minimal friction.

---

##  Key Features (Intended)

| Feature                | Description                                                |
|------------------------|------------------------------------------------------------|
| Image Upload           | Supports JPG, PNG, WEBP formats.                          |
| AI Waste Classification | Identifies waste category and displays results.           |
| Live Feedback          | Provides status updates while loading backend/model.      |
| Responsive Design      | Should adapt well across devices (Smartphones, tablets).  |
| Eco-Focused Mission   | Encourages sustainable waste practices via AI assistance. |

---

##  Development Roadmap

### Completed:
- Frontend UI built with React (or static HTML/CSS/JS).
- Client-side image upload flow with visual feedback.

### In Progress / Needed:
- Functional backend API to receive images and return predictions.
- Integration of `waste_model.h5` (TensorFlow model) via Flask or FastAPI.
- Display AI predictions (category, confidence, recommendations) on the frontend.
- CORS configuration, error handling, and clean UX for failed requests.

---

##  Developer Setup

### Prerequisites:
- Node.js & npm for frontend
- Python 3.10 or 3.11, pip for backend
- Optional: Docker (for containerized deployment)

### Frontend (EcoSave site)
```bash
cd frontend
npm install
npm start
```
### Backend (AI Prediction Service)
```bash
cd backend
python -m venv venv
source venv/bin/activate   # or `.\venv\Scripts\activate` on Windows
pip install -r requirements.txt
python server.py
```

---


### **Contribution Guidelines:**
- Frontend Improvements: Make UX enhancements, refine responsiveness, add loaders or animations.
- Backend Enhancements: Improve error handling, model validation, RESTful structure.
- Testing: Add unit and integration tests (Jest on frontend, pytest on backend).
- Documentation: Keep README up-to-date for future developers and maintainers.

---

## **Created by: EcoSave Team** 
**Atharv Kadam, Yash Dahale, Asad Pathan, Vineet Unde, Shraddha Bhadane**

### **License**
- This project is licensed under the Apache License 2.0 – free to use, modify, and distribute.
- Please give credit to the EcoSave Team when reusing in your projects.
