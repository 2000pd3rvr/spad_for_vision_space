# Machine Vision Plus

A modern web application for real-time object detection using YOLOv8, featuring visitor analytics and geolocation tracking with a sleek dark theme.

## Features

- 🎯 **Real-time Object Detection**: Upload images and get instant object detection results using YOLOv8
- 🌍 **Global Analytics**: Track visitors from around the world with real-time geolocation data
- 📊 **Visitor Statistics**: Monitor usage patterns and visitor demographics
- 🎨 **Modern Dark UI**: Beautiful, responsive interface with dark theme, teal accents, and smooth animations
- 📱 **Mobile Friendly**: Works seamlessly on desktop and mobile devices

## Technology Stack

- **Backend**: Python Flask
- **AI Model**: YOLOv8 (Ultralytics)
- **Database**: SQLite for visitor analytics
- **Frontend**: HTML5, CSS3, JavaScript with Apple Garamond fonts
- **Geolocation**: IP-based location detection

## Installation

1. **Clone or download the project**
   ```bash
   cd /Users/pd3rvr/Documents/object_detection/multiwebapp
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Open your browser**
   Navigate to `http://localhost:5000`

## Project Structure

```
multiwebapp/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── visitors.db           # SQLite database (created automatically)
├── templates/            # HTML templates
│   ├── base.html         # Base template with navigation
│   ├── index.html        # Home page
│   ├── detect.html       # Object detection page
│   └── about.html        # About page
└── static/               # Static assets
    ├── css/
    │   └── style.css     # Main stylesheet
    └── js/
        └── main.js       # JavaScript functionality
```

## Usage

### Home Page
- View visitor statistics and global analytics
- Navigate to detection or about pages
- See real-time visitor counts

### Object Detection
1. Click "Start Detection" or navigate to the detection page
2. Upload an image by dragging and dropping or clicking to browse
3. Click "Detect Objects" to run YOLOv8 analysis
4. View results with bounding boxes and confidence scores

### About Page
- Learn about YOLOv8 and the application features
- View supported object classes
- Understand the technology stack

## Supported Object Classes

YOLOv8 can detect 80 different object categories including:
- People and animals (person, cat, dog, horse, etc.)
- Vehicles (car, truck, bus, motorcycle, etc.)
- Common objects (bottle, cup, laptop, phone, etc.)

## API Endpoints

- `GET /` - Home page
- `GET /detect` - Detection page
- `GET /about` - About page
- `POST /api/detect` - Object detection API
- `GET /api/stats` - Visitor statistics API

## Privacy

- Images are processed in real-time and not stored
- Only anonymous usage statistics are collected (IP, country, city)
- No personal data is stored or transmitted

## Requirements

- Python 3.8+
- Internet connection for geolocation services
- Modern web browser with JavaScript enabled

## Troubleshooting

1. **Model download issues**: The app will automatically download YOLOv8 model on first run
2. **Geolocation not working**: Check internet connection and firewall settings
3. **Image upload errors**: Ensure images are in supported formats (JPEG, PNG, GIF, WebP)

## License

This project is for educational and demonstration purposes.
