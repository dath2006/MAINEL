"""add_thumbnail_base64_to_global_tracks

Revision ID: 378f64073d34
Revises: 
Create Date: 2026-01-07 19:36:47.931796

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
import geoalchemy2


# revision identifiers, used by Alembic.
revision: str = '378f64073d34'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add thumbnail_base64 column to global_tracks table
    op.add_column('global_tracks', sa.Column('thumbnail_base64', sa.Text(), nullable=True))


def downgrade() -> None:
    # Remove thumbnail_base64 column from global_tracks table
    op.drop_column('global_tracks', 'thumbnail_base64')
