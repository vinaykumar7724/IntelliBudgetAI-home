"""utils/chatbot_engine.py — Intent-based chatbot engine for IntelliBudget AI."""
import os
import pickle
import numpy as np
from datetime import datetime

from models import db, Expense, Budget
from utils.nlp import extract_amount_category, extract_description, extract_date
from utils.budget_validator import check_budget_status

# ── Lazy model loading ────────────────────────────────────────────────────────
_model        = None
_tokenizer    = None
_label_encoder = None
MAX_LEN       = 20

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')


def _load_model():
    """Load LSTM model and artefacts once, then cache."""
    global _model, _tokenizer, _label_encoder
    if _model is not None:
        return

    try:
        from tensorflow.keras.models import load_model as keras_load
        _model = keras_load(os.path.join(MODEL_DIR, 'model.h5'))

        with open(os.path.join(MODEL_DIR, 'tokenizer.pkl'), 'rb') as f:
            _tokenizer = pickle.load(f)

        with open(os.path.join(MODEL_DIR, 'label_encoder.pkl'), 'rb') as f:
            _label_encoder = pickle.load(f)

    except Exception as e:
        print(f'[Chatbot] Warning: could not load model — {e}')
        _model = None


def _predict_intent(text: str) -> str:
    """Return the predicted intent label for the given text."""
    _load_model()

    if _model is None:
        return _fallback_intent(text)

    try:
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        seq    = _tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
        probs  = _model.predict(padded, verbose=0)
        idx    = int(np.argmax(probs, axis=1)[0])
        return _label_encoder.inverse_transform([idx])[0]
    except Exception:
        return _fallback_intent(text)


def _fallback_intent(text: str) -> str:
    """Rule-based intent detection when model is unavailable."""
    text_lo = text.lower()

    add_keywords     = ['add', 'spent', 'spend', 'paid', 'pay', 'bought',
                        'expense', 'record', 'log']
    show_keywords    = ['show', 'list', 'view', 'display', 'history', 'transactions']
    analysis_kw      = ['analysis', 'analyse', 'analyze', 'summary', 'report',
                        'how much', 'total']
    salary_kw        = ['salary', 'income', 'earning', 'set salary', 'update salary']
    warning_kw       = ['warning', 'over budget', 'budget', 'exceeded', 'alert']
    greeting_kw      = ['hello', 'hi', 'hey', 'good morning', 'good evening']

    if any(k in text_lo for k in greeting_kw):
        return 'greeting'
    if any(k in text_lo for k in salary_kw):
        return 'set_salary'
    if any(k in text_lo for k in warning_kw):
        return 'warning_query'
    if any(k in text_lo for k in analysis_kw):
        return 'show_analysis'
    if any(k in text_lo for k in show_keywords):
        return 'show_expense'
    if any(k in text_lo for k in add_keywords):
        return 'add_expense'

    import re
    if re.search(r'\b\d+\b', text_lo):
        return 'add_expense'

    return 'unknown'


# ── Intent handlers ───────────────────────────────────────────────────────────

def _handle_greeting(message: str, user) -> str:
    hour = datetime.utcnow().hour
    if hour < 12:
        greeting = 'Good morning'
    elif hour < 17:
        greeting = 'Good afternoon'
    else:
        greeting = 'Good evening'
    return (
        f'{greeting}, {user.username}! 👋\n'
        'I can help you:\n'
        '• Add expenses — "Add 500 to Food"\n'
        '• View expenses — "Show my expenses"\n'
        '• Check budget  — "Am I over budget?"\n'
        '• Set salary    — "Set salary to 50000"'
    )


def _handle_add_expense(message: str, user) -> str:
    amount, category = extract_amount_category(message)
    description      = extract_description(message)

    if amount is None:
        return (
            '❓ I could not find an amount in your message.\n'
            'Try: "Add 500 to Food" or "Spent 200 on transport"'
        )

    # Feature 4: detect date from message
    expense_date = extract_date(message)
    date_str     = expense_date.strftime('%d %b %Y')

    expense = Expense(
        user_id     = user.id,
        amount      = amount,
        category    = category,
        description = description,
        date        = expense_date,
    )
    db.session.add(expense)
    db.session.commit()

    # Budget validation
    budget_status = check_budget_status(user.id, category)

    resp  = f'✅ Added ₹{amount:.2f} to {category} on {date_str}.\n'
    resp += budget_status['message']
    return resp


def _handle_show_expense(message: str, user) -> str:
    today    = datetime.utcnow()
    m_start  = datetime(today.year, today.month, 1)
    expenses = Expense.query.filter(
        Expense.user_id == user.id,
        Expense.date    >= m_start,
    ).order_by(Expense.date.desc()).limit(10).all()

    if not expenses:
        return '📭 No expenses recorded this month yet.'

    lines = ['📋 Your recent expenses this month:\n']
    for e in expenses:
        lines.append(
            f'  • {e.date.strftime("%d %b")} | {e.category} | ₹{e.amount:.2f}'
            + (f' — {e.description}' if e.description else '')
        )
    total = sum(e.amount for e in Expense.query.filter(
        Expense.user_id == user.id,
        Expense.date    >= m_start,
    ).all())
    lines.append(f'\nMonth total so far: ₹{total:.2f}')
    return '\n'.join(lines)


def _handle_show_analysis(message: str, user) -> str:
    today   = datetime.utcnow()
    m_start = datetime(today.year, today.month, 1)
    expenses = Expense.query.filter(
        Expense.user_id == user.id,
        Expense.date    >= m_start,
    ).all()

    if not expenses:
        return '📊 No expenses found for analysis this month.'

    total     = sum(e.amount for e in expenses)
    breakdown = {}
    for e in expenses:
        breakdown[e.category] = breakdown.get(e.category, 0) + e.amount

    salary    = user.monthly_salary or 0
    remaining = salary - total

    lines = [f'📊 Spending Analysis — {today.strftime("%B %Y")}\n']
    lines.append(f'Total spent  : ₹{total:,.2f}')
    if salary:
        lines.append(f'Monthly salary: ₹{salary:,.2f}')
        lines.append(f'Remaining     : ₹{remaining:,.2f}')
    lines.append('\nBy category:')
    for cat, amt in sorted(breakdown.items(), key=lambda x: x[1], reverse=True):
        pct = amt / total * 100 if total else 0
        lines.append(f'  {cat:<15} ₹{amt:>8.2f}  ({pct:.1f}%)')
    return '\n'.join(lines)


def _handle_set_salary(message: str, user) -> str:
    import re
    match = re.search(r'(\d[\d,]*(?:\.\d{1,2})?)', message)
    if not match:
        return '❓ Please include the salary amount, e.g. "Set salary to 50000"'

    amount = float(match.group(1).replace(',', ''))
    user.monthly_salary = amount
    db.session.commit()
    return f'✅ Monthly salary updated to ₹{amount:,.2f}.'


def _handle_warning_query(message: str, user) -> str:
    from utils.budget_validator import get_warned_categories
    warnings = get_warned_categories(user.id)

    if not warnings:
        return '✅ All your budgets are within safe limits!'

    lines = [f'⚠️ Budget alerts ({len(warnings)} categories):\n']
    for w in warnings:
        lines.append(f'  • {w["message"]}')
    return '\n'.join(lines)


def _handle_unknown(message: str, user) -> str:
    return (
        "🤔 I didn't quite understand that.\n"
        'You can say things like:\n'
        '  • "Add 300 to Food"\n'
        '  • "Show my expenses"\n'
        '  • "How much did I spend this month?"\n'
        '  • "Am I over budget?"\n'
        '  • "Set salary to 40000"'
    )


# ── Main dispatcher ───────────────────────────────────────────────────────────

_HANDLERS = {
    'greeting':      _handle_greeting,
    'add_expense':   _handle_add_expense,
    'show_expense':  _handle_show_expense,
    'show_analysis': _handle_show_analysis,
    'set_salary':    _handle_set_salary,
    'warning_query': _handle_warning_query,
}


class Chatbot:
    """Main chatbot interface used by Flask routes."""

    def handle_message(self, message: str, user) -> str:
        if not message or not message.strip():
            return '💬 Please type a message.'

        intent  = _predict_intent(message.strip())
        handler = _HANDLERS.get(intent, _handle_unknown)
        return handler(message.strip(), user)
