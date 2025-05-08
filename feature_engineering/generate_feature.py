import pandas as pd
import random
from datetime import datetime, timedelta
from typing import List, Dict
import numpy as np
import re
from faker import Faker

# Load existing data
data = pd.read_csv(r"C:\Users\pc\Desktop\student dataset\augmented dataset\NEW_enhanced_student_complaints.csv")

class ComplaintDataGenerator:
    def __init__(self, existing_df: pd.DataFrame = None):
        self.faker = Faker()
        self.existing_df = existing_df
        
        # Categories without subcategories
        self.categories = [
            'International student experiences',
            'Online learning',
            'Student Affairs',
            'Housing and Transportation',
            'Activities and Travelling',
            'Food and Cantines',
            'Academic Support and Resources',
            'Athletics and sports',
            'Career opportunities',
            'Financial Support',
            'Health and Well-being Support'
        ]
        
        self.category_response_profiles = {
            'International student experiences': {
                'base': 48, 'variation': 24, 'urgency_factor': 0.7, 'resolution_variation': (1, 3)
            },
            'Online learning': {
                'base': 24, 'variation': 12, 'urgency_factor': 0.8, 'resolution_variation': (1, 2)
            },
            'Student Affairs': {
                'base': 36, 'variation': 18, 'urgency_factor': 0.6, 'resolution_variation': (1, 2.5)
            },
            'Housing and Transportation': {
                'base': 72, 'variation': 48, 'urgency_factor': 0.9, 'resolution_variation': (1, 4)
            },
            'Activities and Travelling': {
                'base': 60, 'variation': 36, 'urgency_factor': 0.5, 'resolution_variation': (1, 3)
            },
            'Food and Cantines': {
                'base': 12, 'variation': 6, 'urgency_factor': 0.8, 'resolution_variation': (1, 2)
            },
            'Academic Support and Resources': {
                'base': 24, 'variation': 12, 'urgency_factor': 0.7, 'resolution_variation': (1, 2)
            },
            'Athletics and sports': {
                'base': 48, 'variation': 24, 'urgency_factor': 0.6, 'resolution_variation': (1, 3)
            },
            'Career opportunities': {
                'base': 96, 'variation': 48, 'urgency_factor': 0.4, 'resolution_variation': (1, 5)
            },
            'Financial Support': {
                'base': 120, 'variation': 72, 'urgency_factor': 0.9, 'resolution_variation': (1, 6)
            },
            'Health and Well-being Support': {
                'base': 6, 'variation': 3, 'urgency_factor': 1.0, 'resolution_variation': (1, 1.5)
            }
        }
        
        self.priorities = ['low', 'medium', 'high', 'critical']
        self.priority_factors = {'low': 1.5, 'medium': 1.0, 'high': 0.7, 'critical': 0.3}
        
        self.status_options = {
            'pending': 0.2,
            'in_progress': 0.3,
            'resolved': 0.4,
            'rejected': 0.1,
        }
    
    def _generate_response_time(self, category: str, priority: str) -> dict:
        profile = self.category_response_profiles[category]
        
        base_hours = np.random.lognormal(
            mean=np.log(profile['base']),
            sigma=0.2
        )
        
        response_hours = (base_hours * self.priority_factors[priority] * profile['urgency_factor']) + \
                        random.uniform(-profile['variation'], profile['variation'])
        response_hours = max(1, min(response_hours, 720))  # 1 hour to 30 days
        
        return {
            'response_hours': round(response_hours, 2),
            'response_delta': timedelta(hours=response_hours),
            'resolution_variation': profile['resolution_variation']
        }
    
    def generate_complaints(self, num_complaints: int = None) -> List[Dict]:
        complaints = []
        
        # Determine number of complaints based on input
        if self.existing_df is not None:
            num_complaints = len(self.existing_df)
            categories = self.existing_df['Genre'].tolist()
        else:
            num_complaints = num_complaints if num_complaints else 100
            categories = [random.choice(self.categories) for _ in range(num_complaints)]
        
        for i in range(num_complaints):
            # Use existing category if available, otherwise random
            category = categories[i] if self.existing_df is not None else random.choice(self.categories)
            priority = random.choices(self.priorities, weights=[0.3, 0.4, 0.2, 0.1])[0]
            
            # Realistic date distribution (more recent complaints more likely)
            days_ago = min(int(np.random.gamma(shape=2, scale=30)), 180)
            report_time = datetime.now() - timedelta(days=days_ago)
            
            # Business hours (9am-5pm)
            report_time = report_time.replace(
                hour=random.randint(9, 17),
                minute=random.randint(0, 59),
                second=random.randint(0, 59))
            
            time_data = self._generate_response_time(category, priority)
            status = random.choices(list(self.status_options.keys()), weights=list(self.status_options.values()))[0]
            
            # Modified logic to ensure all statuses have response times
            if status in ['resolved', 'rejected']:
                # For resolved/rejected, use full response time
                response_time = report_time + time_data['response_delta']
                response_days = round(time_data['response_delta'].total_seconds() / 86400, 2)
            elif status == 'in_progress':
                # For in_progress, use partial response time (25-75% of total)
                progress_factor = random.uniform(0.25, 0.75)
                partial_delta = timedelta(hours=time_data['response_hours'] * progress_factor)
                response_time = report_time + partial_delta
                response_days = round(partial_delta.total_seconds() / 86400, 2)
            else:  # pending
                # For pending, use initial response time (1-25% of total)
                pending_factor = random.uniform(0.01, 0.25)
                pending_delta = timedelta(hours=time_data['response_hours'] * pending_factor)
                response_time = report_time + pending_delta
                response_days = round(pending_delta.total_seconds() / 86400, 2)
            
            complaint = {
                'category': category,
                'priority': priority,
                'report_time': report_time.strftime('%Y-%m-%d %H:%M'),
                'response_time': response_time.strftime('%Y-%m-%d %H:%M'),
                'response_days': response_days,
                'status': status,
                'report_hour': report_time.hour
            }
            
            complaints.append(complaint)
        
        return complaints
    
    def generate_dataframe(self, num_complaints: int = None) -> pd.DataFrame:
        complaints = self.generate_complaints(num_complaints)
        df = pd.DataFrame(complaints)
        
        # Clean column order and handling
        final_columns = [
            'category', 'priority', 'status',
            'report_time', 'report_hour',
            'response_time', 'response_days'
        ]
        
        return df[final_columns]


def generate_enhanced_email(df, name_column, domain="Gmail.com"):
    """
    Generate enhanced email addresses from student names in a DataFrame.
    
    
    pd.DataFrame: Original DataFrame with an added 'email' column
    """
    
    def process_name(name):
        # Clean the name - remove extra spaces and special characters
        name = re.sub(r'[^a-zA-Z\s]', '', name).strip()
        parts = name.split()
        
        if len(parts) == 0:
            return ""
        
        # Get first and last name
        first = parts[0].lower()
        last = parts[-1].lower() if len(parts) > 1 else ""
        
        # Generate email (using first.last format)
        if last:
            email = f"{first}.{last}@{domain}"
        else:
            email = f"{first}@{domain}"
        
        return email
    
    # Apply to each name
    df['Gmail'] = df[name_column].apply(lambda x: process_name(x) if pd.notna(x) else "")
    
    return df


class IDGenerator:
    def __init__(self, existing_tickets: List[str] = None, existing_students: List[str] = None):
        self.existing_tickets = set(existing_tickets) if existing_tickets else set()
        self.existing_students = set(existing_students) if existing_students else set()
        
    def _generate_ticket_id(self) -> str:
        """Generates a unique TKT-ID in format TKT-######"""
        while True:
            ticket_num = f"{random.randint(0, 999999):06d}"  # 6-digit zero-padded
            ticket_id = f"TKT-{ticket_num}"
            if ticket_id not in self.existing_tickets:
                self.existing_tickets.add(ticket_id)
                return ticket_id
    
    def _generate_student_id(self) -> str:
        """Generates a unique STU-ID in format STU-#####"""
        while True:
            student_num = f"{random.randint(0, 99999):05d}"  # 5-digit zero-padded
            student_id = f"STU-{student_num}"
            if student_id not in self.existing_students:
                self.existing_students.add(student_id)
                return student_id
    
    def add_ids_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds Ticket ID and Student ID columns to DataFrame"""
        df = df.copy()
        df["Ticket ID"] = [self._generate_ticket_id() for _ in range(len(df))]
        df["Student_ID"] = [self._generate_student_id() for _ in range(len(df))]
        return df

# Example Usage
if __name__ == "__main__":
    


        # Example usage with existing dataframe
    generator = ComplaintDataGenerator(existing_df=data)
    complaints_df = generator.generate_dataframe()
    
    # Clean NaN values
    complaints_df.fillna({
        'response_time': 'Pending',
        'response_days': 'N/A'
    }, inplace=True)
    
    # Print statistics
    print("\nComplaint Status Distribution:")
    print(complaints_df['status'].value_counts(normalize=True).round(3))
    
    print("\nAverage Response Times by Category (days):")
    avg_response = complaints_df[complaints_df['response_days'] != 'N/A']\
        .groupby('category')['response_days'].mean()\
        .sort_values().round(1)
    print(avg_response)
    
    # Drop the category column as it duplicates the 'Genre' column
    complaints_df.drop("category", axis=1, inplace=True)
    
    # Combine with original data
    alldata = pd.concat([data, complaints_df], axis=1)
    
    # Print sample of the final data
    print("\nFinal combined data sample:")
    print(alldata.columns)
    display=['Genre',
        'priority', 'status', 'report_time', 'report_hour', 'response_time',
        'response_days']
    print(alldata[display].tail(15).to_string(index=False))

    # Generate emails
    enhanced_df = generate_enhanced_email(alldata, 'Student Name', "Gmail.com")
    
    # print(enhanced_df[['Student Name', 'Gmail']])
    
    id_gen = IDGenerator()
    enhanced_df =id_gen.add_ids_to_dataframe(enhanced_df)
    print(enhanced_df)
    print(enhanced_df.columns)

    print(enhanced_df.isna().sum())

    print(enhanced_df.duplicated().sum())


    col_list=["Ticket ID","Student_ID"]
    print(f"the duplicated columns \n" +"\n".join(f"{col}:{enhanced_df[col].duplicated().sum()}" for col in col_list))

    enhanced_df.to_csv(r"C:\Users\pc\Desktop\student dataset\augmented dataset\Final_EnhancedGmail.csv",index=False)



