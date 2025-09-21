# Contributing Guide

## Welcome!

ขอบคุณที่สนใจจะช่วยพัฒนาระบบ **CNN Model Deployment**! คู่มือนี้จะช่วยให้คุณสามารถมีส่วนร่วมในการพัฒนาระบบได้อย่างมีประสิทธิภาพ

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
# Contributing Guide
- [Feature Requests](#feature-requests)
- [Documentation](#documentation)
- [Testing](#testing)
- [Deployment](#deployment)

## Getting Started

### **Prerequisites**
- **Python:** 3.8+ (แนะนำ 3.9+)
- **Git:** สำหรับ version control
- **GitHub Account:** สำหรับ fork และ pull request
- **Basic Knowledge:** Python, FastAPI, TensorFlow

### **Fork Repository**
1. ไปที่ [GitHub Repository](https://github.com/yourusername/cnn-model-deployment)
2. คลิกปุ่ม "Fork" มุมขวาบน
3. Clone repository ที่ fork มา:
   ```bash
   git clone https://github.com/yourusername/cnn-model-deployment.git
   cd cnn-model-deployment
   ```

### **Setup Development Environment**
```bash
# สร้าง virtual environment
python -m venv venv

# เปิดใช้งาน (Windows)
venv\Scripts\activate

# เปิดใช้งาน (macOS/Linux)
source venv/bin/activate

# ติดตั้ง dependencies
pip install -r requirements.txt

# ติดตั้ง development dependencies
pip install -r requirements-dev.txt
```

## Development Setup

### **Project Structure**
```
cnn-model-deployment/
├── app/
│   ├── main.py              # FastAPI application
│   ├── inference.py         # AI model logic
│   └── templates/
│       └── index.html       # Frontend UI
├── models/                  # AI models (.keras files)
├── docs/                    # Documentation
├── tests/                   # Test files
├── requirements.txt         # Production dependencies
├── requirements-dev.txt     # Development dependencies
├── .gitignore              # Git ignore rules
├── .pre-commit-config.yaml # Pre-commit hooks
└── README.md               # Main documentation
```

### **Development Dependencies**
```txt
# requirements-dev.txt
pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-cov>=4.0.0
black>=23.0.0
isort>=5.12.0
flake8>=6.0.0
mypy>=1.0.0
pre-commit>=3.0.0
```

### **Pre-commit Hooks**
```bash
# ติดตั้ง pre-commit hooks
pre-commit install

# รัน hooks แบบ manual
pre-commit run --all-files
```

## Code Style

### **Python Code Style**

#### **Black (Code Formatter)**
```bash
# รัน Black
black app/ tests/

# ตรวจสอบโดยไม่แก้ไข
black --check app/ tests/
```

#### **isort (Import Sorter)**
```bash
# รัน isort
isort app/ tests/

# ตรวจสอบโดยไม่แก้ไข
isort --check-only app/ tests/
```

#### **Flake8 (Linter)**
```bash
# รัน Flake8
flake8 app/ tests/

# รันกับ configuration
flake8 --config .flake8 app/ tests/
```

#### **MyPy (Type Checker)**
```bash
# รัน MyPy
mypy app/

# รันกับ configuration
mypy --config-file mypy.ini app/
```

### **Code Style Guidelines**

#### **1. Python**
```python
# ✅ Good
def predict_from_ndarray(img: np.ndarray, enable_detection: bool = True) -> Dict[str, Any]:
    """
    Predict from numpy array.
    
    Args:
        img: Input image as numpy array
        enable_detection: Whether to enable object detection
        
    Returns:
        Dictionary containing prediction results
    """
    try:
        model = load_model()
        result = model.predict(img)
        return {"ok": True, "result": result}
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return {"ok": False, "error": str(e)}

# ❌ Bad
def predict(img, enable_detection=True):
    try:
        model = load_model()
        result = model.predict(img)
        return {"ok": True, "result": result}
    except Exception as e:
        return {"ok": False, "error": str(e)}
```

#### **2. HTML**
```html
<!-- ✅ Good -->
<div class="upload-area" id="upload-area">
    <div class="upload-content">
        <svg class="upload-icon" width="48" height="48" viewBox="0 0 24 24">
            <path d="M14,2H6A2,2 0 0,0 4,4V20A2,2 0 0,0 6,22H18A2,2 0 0,0 20,20V8L14,2M18,20H6V4H13V9H18V20Z" />
        </svg>
        <h3>ลากไฟล์มาวางที่นี่</h3>
        <p>หรือคลิกเพื่อเลือกไฟล์</p>
    </div>
    <input type="file" id="file-input" accept="image/*" hidden>
</div>

<!-- ❌ Bad -->
<div id="upload-area"><div class="upload-content"><svg class="upload-icon"><path d="M14,2H6A2,2 0 0,0 4,4V20A2,2 0 0,0 6,22H18A2,2 0 0,0 20,20V8L14,2M18,20H6V4H13V9H18V20Z"/></svg><h3>ลากไฟล์มาวางที่นี่</h3><p>หรือคลิกเพื่อเลือกไฟล์</p></div><input type="file" id="file-input" accept="image/*" hidden></div>
```

#### **3. CSS**
```css
/* ✅ Good */
.upload-area {
    border: 2px dashed var(--primary-blue);
    border-radius: var(--radius-lg);
    padding: var(--spacing-8);
    text-align: center;
    cursor: pointer;
    transition: all 0.3s ease;
    background: var(--white);
}

.upload-area:hover {
    border-color: var(--secondary-blue);
    background: var(--light-blue);
}

/* ❌ Bad */
.upload-area{border:2px dashed #2563eb;border-radius:0.75rem;padding:2rem;text-align:center;cursor:pointer;transition:all 0.3s ease;background:#ffffff}.upload-area:hover{border-color:#3b82f6;background:#dbeafe}
```

#### **4. JavaScript**
```javascript
// ✅ Good
class CNNModelApp {
    constructor() {
        this.devMode = false;
        this.currentModel = 'MobileNetV2_Augmented_best.keras';
        this.performanceMonitor = new PerformanceMonitor();
        
        this.initializeEventListeners();
        this.loadAvailableModels();
    }

    async processFile(file) {
        try {
            this.showProgress();
            
            const formData = new FormData();
            formData.append('file', file);
            formData.append('model_name', this.currentModel);
            
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (result.ok) {
                this.displayResults(result);
            } else {
                this.showError(result.error);
            }
        } catch (error) {
            this.showError('เกิดข้อผิดพลาดในการประมวลผล');
        } finally {
            this.hideProgress();
        }
    }
}

// ❌ Bad
class CNNModelApp{constructor(){this.devMode=false;this.currentModel='MobileNetV2_Augmented_best.keras';this.performanceMonitor=new PerformanceMonitor();this.initializeEventListeners();this.loadAvailableModels();}async processFile(file){try{this.showProgress();const formData=new FormData();formData.append('file',file);formData.append('model_name',this.currentModel);const response=await fetch('/predict',{method:'POST',body:formData});const result=await response.json();if(result.ok){this.displayResults(result);}else{this.showError(result.error);}}catch(error){this.showError('เกิดข้อผิดพลาดในการประมวลผล');}finally{this.hideProgress();}}}
```

## Pull Request Process

### **1. Create Feature Branch**
```bash
# สร้าง branch ใหม่
git checkout -b feature/amazing-feature

# หรือ
git checkout -b fix/bug-fix
```

### **2. Make Changes**
- แก้ไขโค้ดตามที่ต้องการ
- เพิ่ม tests สำหรับ features ใหม่
- อัปเดต documentation
- รัน tests และ linting

### **3. Commit Changes**
```bash
# Add changes
git add .

# Commit with descriptive message
git commit -m "feat: add new model support

- Add EfficientNetV2B0 model
- Update model selection UI
- Add model performance metrics
- Update documentation

Closes #123"
```

### **Commit Message Format**
```
<type>(<scope>): <description>

<body>

<footer>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

**Examples:**
```bash
feat(ui): add dark mode support
fix(api): resolve model loading error
docs(readme): update installation guide
style(css): improve button hover effects
refactor(inference): optimize image processing
test(api): add prediction endpoint tests
chore(deps): update tensorflow to 2.13.0
```

### **4. Push Changes**
```bash
git push origin feature/amazing-feature
```

### **5. Create Pull Request**
1. ไปที่ GitHub repository
2. คลิก "Compare & pull request"
3. กรอกข้อมูล:
   - **Title:** ชื่อที่อธิบายการเปลี่ยนแปลง
   - **Description:** รายละเอียดการเปลี่ยนแปลง
   - **Labels:** เลือก label ที่เหมาะสม
   - **Assignees:** กำหนดผู้รับผิดชอบ
   - **Reviewers:** กำหนดผู้ review

### **6. Pull Request Template**
```markdown
## 📝 Description
Brief description of changes

## 🔗 Related Issues
Closes #123
Fixes #456

## 🧪 Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Performance tests pass

## 📸 Screenshots
(If applicable)

## 📋 Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] No breaking changes
```

## Issue Reporting

### **Bug Report Template**
```markdown
## Bug Description
Clear description of the bug

## Steps to Reproduce
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

## Expected Behavior
What you expected to happen

## Actual Behavior
What actually happened

## Screenshots
(If applicable)

## Environment
- OS: [e.g. Windows 10, macOS 12, Ubuntu 20.04]
- Python: [e.g. 3.9.7]
- Browser: [e.g. Chrome 91, Firefox 89]
- Version: [e.g. 1.0.0]

## Additional Context
Any other context about the problem
```

### **Feature Request Template**
```markdown
## Feature Description
Clear description of the feature

## Motivation
Why is this feature needed?

## Detailed Requirements
- Requirement 1
- Requirement 2
- Requirement 3

## UI/UX Mockups
(If applicable)

## Related Issues
Links to related issues or discussions

## Additional Context
Any other context about the feature
```

## Testing

### **Running Tests**
```bash
# รัน tests ทั้งหมด
pytest

# รัน tests พร้อม coverage
pytest --cov=app --cov-report=html

# รัน tests แบบ verbose
pytest -v

# รัน tests เฉพาะไฟล์
pytest tests/test_api.py

# รัน tests เฉพาะ function
pytest tests/test_api.py::test_predict
```

### **Test Structure**
```
tests/
├── __init__.py
├── conftest.py          # Test configuration
├── test_api.py          # API tests
├── test_inference.py    # Inference tests
├── test_models.py       # Model tests
└── test_ui.py           # UI tests
```

### **Test Examples**
```python
# test_api.py
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_predict_success():
    """Test successful prediction"""
    with open("tests/test_image.jpg", "rb") as f:
        response = client.post("/predict", files={"file": f})
    
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] == True
    assert "detections" in data

def test_predict_invalid_file():
    """Test prediction with invalid file"""
    response = client.post("/predict", files={"file": ("test.txt", "content", "text/plain")})
    
    assert response.status_code == 422

def test_available_models():
    """Test available models endpoint"""
    response = client.get("/available-models")
    
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert len(data["models"]) > 0
```

## Documentation

### **Documentation Standards**
- **ภาษาไทย** - ใช้ภาษาไทยสำหรับ user-facing documentation
- **ภาษาอังกฤษ** - ใช้ภาษาอังกฤษสำหรับ technical documentation
- **Markdown** - ใช้ Markdown format
- **Clear Structure** - โครงสร้างชัดเจน
- **Examples** - มีตัวอย่างการใช้งาน

### **Documentation Files**
```
docs/
├── README.md                    # Main documentation
├── API_DOCUMENTATION.md         # API documentation
├── INSTALLATION_GUIDE.md        # Installation guide
├── USER_GUIDE.md                # User guide
├── DEVELOPER_GUIDE.md           # Developer guide
├── CONTRIBUTING.md              # Contributing guide
├── CHANGELOG.md                 # Changelog
└── TROUBLESHOOTING.md           # Troubleshooting guide
```

### **Documentation Guidelines**
- **Clear Headers** - ใช้ header ที่ชัดเจน
- **Code Examples** - มีตัวอย่างโค้ด
- **Screenshots** - มีภาพประกอบ
- **Links** - มีลิงก์ที่เกี่ยวข้อง
- **Table of Contents** - มีสารบัญ

## Deployment

### **Testing Deployment**
```bash
# Test local deployment
python -m uvicorn app.main:app --reload

# Test Docker deployment
docker build -t cnn-model-deployment .
docker run -p 8000:8000 cnn-model-deployment

# Test production deployment
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

### **Deployment Checklist**
- [ ] All tests pass
- [ ] Documentation updated
- [ ] Version bumped
- [ ] Changelog updated
- [ ] Docker image built
- [ ] Production deployment tested

## Code Review Guidelines

### **For Reviewers**
- **Be Constructive** - ให้คำแนะนำที่เป็นประโยชน์
- **Be Respectful** - ใช้คำพูดที่สุภาพ
- **Be Specific** - ระบุปัญหาให้ชัดเจน
- **Be Timely** - review ภายใน 48 ชั่วโมง

### **For Authors**
- **Be Responsive** - ตอบกลับ feedback อย่างรวดเร็ว
- **Be Open** - เปิดรับคำแนะนำ
- **Be Patient** - รอการ review อย่างอดทน
- **Be Grateful** - ขอบคุณผู้ review

## Contribution Areas

### **Code Contributions**
- **Bug Fixes** - แก้ไข bugs
- **New Features** - เพิ่ม features ใหม่
- **Performance Improvements** - ปรับปรุงประสิทธิภาพ
- **Code Refactoring** - ปรับปรุงโค้ด

### **Documentation Contributions**
- **User Guides** - คู่มือผู้ใช้
- **API Documentation** - เอกสาร API
- **Tutorials** - บทความสอน
- **Translations** - แปลภาษา

### **Testing Contributions**
- **Unit Tests** - ทดสอบแต่ละส่วน
- **Integration Tests** - ทดสอบการทำงานร่วมกัน
- **Performance Tests** - ทดสอบประสิทธิภาพ
- **UI Tests** - ทดสอบ UI

### **Design Contributions**
- **UI/UX Improvements** - ปรับปรุง UI/UX
- **Icons** - ไอคอนใหม่
- **Color Schemes** - โทนสีใหม่
- **Layouts** - เลย์เอาต์ใหม่

## Recognition

### **Contributor Levels**
- **Bronze** - 1-5 contributions
- **Silver** - 6-15 contributions
- **Gold** - 16-30 contributions
- **Diamond** - 31+ contributions

### **Special Recognition**
- **Star Contributor** - Outstanding contributions
- **Bug Hunter** - Excellent bug reports
- **Documentation Hero** - Great documentation
- **Testing Champion** - Comprehensive testing

## Getting Help

### **Communication Channels**
- **GitHub Issues** - สำหรับ bugs และ features
- **GitHub Discussions** - สำหรับคำถามทั่วไป
- **Email** - your-email@example.com
- **Discord** - [Discord Server](https://discord.gg/your-server)

### **Response Times**
- **Critical Bugs** - 24 hours
- **Regular Issues** - 3-5 days
- **Feature Requests** - 1-2 weeks
- **General Questions** - 1-3 days

## License

โปรเจคนี้ใช้ MIT License - ดูรายละเอียดใน [LICENSE](LICENSE) file

## Thank You!

ขอบคุณที่สนใจจะช่วยพัฒนาระบบ **CNN Model Deployment**!

การมีส่วนร่วมของคุณจะช่วยให้ระบบดีขึ้นและเป็นประโยชน์ต่อผู้ใช้มากขึ้น

**Happy Contributing!**

---

## Additional Resources

### **Learning Resources**
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Python Documentation](https://docs.python.org/)
- [Git Documentation](https://git-scm.com/doc)

### **Development Tools**
- [VS Code](https://code.visualstudio.com/)
- [PyCharm](https://www.jetbrains.com/pycharm/)
- [Docker](https://www.docker.com/)
- [Postman](https://www.postman.com/)

### **Community**
- [Python Reddit](https://www.reddit.com/r/Python/)
- [FastAPI Discord](https://discord.gg/9Z9CgHQ)
- [TensorFlow Forum](https://discuss.tensorflow.org/)
- [Stack Overflow](https://stackoverflow.com/)

**สู้ๆ ครับ!**
