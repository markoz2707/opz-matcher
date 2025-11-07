"""
Utility script to create an admin user
"""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

from sqlalchemy import select
from passlib.context import CryptContext

from services.database import AsyncSessionLocal, init_db
from models.database import User


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


async def create_admin_user(
    username: str,
    email: str,
    password: str,
    full_name: str = None
):
    """Create an admin user"""
    
    # Initialize database
    await init_db()
    
    async with AsyncSessionLocal() as db:
        # Check if user exists
        result = await db.execute(
            select(User).where(
                (User.username == username) | (User.email == email)
            )
        )
        existing_user = result.scalar_one_or_none()
        
        if existing_user:
            print(f"Error: User with username '{username}' or email '{email}' already exists")
            return False
        
        # Hash password
        hashed_password = pwd_context.hash(password)
        
        # Create user
        user = User(
            username=username,
            email=email,
            hashed_password=hashed_password,
            full_name=full_name,
            is_active=True,
            is_superuser=True
        )
        
        db.add(user)
        await db.commit()
        
        print(f"✓ Admin user created successfully:")
        print(f"  Username: {username}")
        print(f"  Email: {email}")
        print(f"  Superuser: Yes")
        
        return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create an admin user")
    parser.add_argument("--username", required=True, help="Username")
    parser.add_argument("--email", required=True, help="Email address")
    parser.add_argument("--password", required=True, help="Password")
    parser.add_argument("--full-name", help="Full name", default=None)
    
    args = parser.parse_args()
    
    asyncio.run(create_admin_user(
        username=args.username,
        email=args.email,
        password=args.password,
        full_name=args.full_name
    ))
