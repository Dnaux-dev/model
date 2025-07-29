from sqlalchemy import Column, Integer, String, ForeignKey, Float
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class Owner(Base):
    __tablename__ = 'owners'
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    tagged_objects = relationship('TaggedObject', back_populates='owner')

class TaggedObject(Base):
    __tablename__ = 'tagged_objects'
    id = Column(Integer, primary_key=True)
    object_class = Column(String, nullable=False)
    bbox = Column(String, nullable=False)  # Store as JSON string
    image_path = Column(String, nullable=False)
    owner_id = Column(Integer, ForeignKey('owners.id'))
    owner = relationship('Owner', back_populates='tagged_objects') 